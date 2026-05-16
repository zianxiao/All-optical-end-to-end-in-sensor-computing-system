# -*- coding: utf-8 -*-

import time
import socket
import ctypes
from ctypes import Structure, c_ubyte, c_ushort, c_double, sizeof, addressof

import nidaqmx
from nidaqmx.constants import AcquisitionType


# ----------------------------
# MCP packet struct (8 bytes)
# ----------------------------
class MCPSEND(Structure):
    _pack_ = 1
    _fields_ = [("data", c_ubyte * 8)]


def _bytes_from_send(pkt: MCPSEND) -> bytes:
    """Convert MCPSEND structure to bytes reliably for UDP transmission."""
    return ctypes.string_at(addressof(pkt), sizeof(pkt))


# ----------------------------
# NI-9215: Read current state (actual weight w)
# ----------------------------
def ni9215_read_w(ni_physical_ch: str) -> float:
    """Read 5 samples and calculate the mean as the current feedback voltage/weight."""
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(
            ni_physical_ch,
            min_val=0.0,
            max_val=10.0
        )
        task.timing.cfg_samp_clk_timing(
            rate=1000,
            sample_mode=AcquisitionType.FINITE,
            samps_per_chan=5
        )
        data = task.read(number_of_samples_per_channel=5)
    return sum(data) / len(data)


# ----------------------------
# ChiL Negative Feedback Control Main Function
# ----------------------------
def chil_feedback_control(
    target_w: float,               # Target feedback value to achieve (W_hat)
    dll_path=r".\MCP.dll",
    dac_ip="169.254.1.10",
    dac_port=1234,
    dac_channel=39,
    dac_cfg_param=16.284,
    ni_physical_ch="cDAQ9185-2395476Mod1/ai0",
    max_iter=15,                   # Max iterations (~10 iterations typically needed for convergence)
    learning_rate=0.5,             # Learning rate eta (0.5 is recommended by the paper)
    settle_s=0.01,                 # Settle time for hardware response
    v_min=0.0,                     # Minimum safe DAC voltage limit
    v_max=1.0,                     # Maximum safe DAC voltage limit
):
    # 1. Load C-DLL
    lib = ctypes.cdll.LoadLibrary(dll_path)
    lib.makeSetProtocol.argtypes = [c_ushort, c_double, c_double]
    lib.makeSetProtocol.restype = MCPSEND

    # 2. Initialize UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (dac_ip, dac_port)

    print("=" * 50)
    print("Starting ChiL Adaptive Closed-Loop Control")
    print(f"Target W: {target_w:.4f} V")
    print(f"DAC Channel: {dac_channel} | Feedback NI Channel: {ni_physical_ch}")
    print("=" * 50)

    try:
        # ---- Step 0: Initial Guess (Iteration 0) ----
        p_k = 0.2  # Initial guess voltage 0.2V
        pkt = lib.makeSetProtocol(dac_channel, float(p_k), float(dac_cfg_param))
        sock.sendto(_bytes_from_send(pkt), addr)
        time.sleep(settle_s)
        w_k = ni9215_read_w(ni_physical_ch)
        
        print(f"Iter  0 | DAC V = {p_k:.4f} V -> Measured W = {w_k:.4f} V | Error = {(w_k - target_w):.4f}")

        # ---- Step 1: Single Perturbation Step to Establish Direction (Iteration 1) ----
        p_prev = p_k
        w_prev = w_k
        
        # Generate a small step displacement
        p_k = p_prev + 0.05 if p_prev + 0.05 <= v_max else p_prev - 0.05
        pkt = lib.makeSetProtocol(dac_channel, float(p_k), float(dac_cfg_param))
        sock.sendto(_bytes_from_send(pkt), addr)
        time.sleep(settle_s)
        w_k = ni9215_read_w(ni_physical_ch)
        
        print(f"Iter  1 | DAC V = {p_k:.4f} V -> Measured W = {w_k:.4f} V | Error = {(w_k - target_w):.4f}")

        # ---- Step 2 to max_iter: Closed-Loop Adaptive Feedback Iterations ----
        for k in range(2, max_iter + 1):
            # Calculate current error
            error = w_k - target_w
            
            if abs(error) < 0.002:  # Set a reasonable convergence tolerance (deadband)
                print(f"--> Target reached early! Convergence criteria met.")
                break

            # Calculate historical finite difference derivative: f_prime = delta_w / delta_p
            delta_w = w_k - w_prev
            delta_p = p_k - p_prev
            
            if abs(delta_p) > 1e-6:
                f_prime = delta_w / delta_p
            else:
                f_prime = 1.0  # Prevent division by zero

            # [Core Optimization] Gradient Clipping to prevent step size explosion or direction reversal
            sign = 1.0 if f_prime >= 0 else -1.0
            # Constrain the absolute derivative value between [0.2, 20.0]
            f_prime_clipped = sign * max(0.2, min(abs(f_prime), 20.0))

            # Update the next voltage value based on the Quasi-Newton ChiL rule
            # p_next = p_k - learning_rate * error / f_prime
            delta_p_next = - (learning_rate * error) / f_prime_clipped
            
            # Limit maximum single-step voltage jump to prevent severe hardware oscillation
            max_step = 0.15
            delta_p_next = max(-max_step, min(delta_p_next, max_step))

            p_next = p_k + delta_p_next
            # Enforce safe boundary limits
            p_next = max(v_min, min(p_next, v_max))

            # Backup history for the next iteration's finite difference
            p_prev, w_prev = p_k, w_k

            # Apply to hardware
            p_k = p_next
            pkt = lib.makeSetProtocol(dac_channel, float(p_k), float(dac_cfg_param))
            sock.sendto(_bytes_from_send(pkt), addr)
            
            # Wait for hardware to settle and read back the feedback value
            time.sleep(settle_s)
            w_k = ni9215_read_w(ni_physical_ch)

            print(f"Iter {k:2d} | DAC V = {p_k:.4f} V -> Measured W = {w_k:.4f} V | Error = {error:.4f} | f'={f_prime:.2f}")

    except Exception as e:
        print(f"An exception occurred during control execution: {e}")

    finally:
        # Safe Exit: Reset DAC channel safely back to 0V
        try:
            pkt_end = lib.makeSetProtocol(dac_channel, 0.0, float(dac_cfg_param))
            sock.sendto(_bytes_from_send(pkt_end), addr)
            print("\n[Safe Exit] DAC channel safely reset to 0V.")
        except Exception as e:
            print(f"WARNING: DAC reset failed: {e}")
        sock.close()

    print("Done.")


if __name__ == "__main__":
    # Example: Run closed-loop feedback control to maintain NI readback average voltage at 0.45V
    chil_feedback_control(
        target_w=0.45,
        dac_channel=39,
        ni_physical_ch="cDAQ9185-2395476Mod1/ai0",
        max_iter=15,
        learning_rate=0.5
    )
