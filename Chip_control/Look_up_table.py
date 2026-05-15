# -*- coding: utf-8 -*-

import time
import csv
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
    """把 MCPSEND 结构体稳定地转成 bytes 以便 UDP 发送。"""
    return ctypes.string_at(addressof(pkt), sizeof(pkt))


# ----------------------------
# NI-9215: read 5 samples mean (你的验证版本)
# ----------------------------
def ni9215_read_5pt_mean(ni_physical_ch: str) -> float:
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
# Main: sweep DAC 0~0.5V and log CSV
# ----------------------------
def sweep_0_to_0p5V_and_log(
    dll_path=r".\MCP.dll",
    dac_ip="169.254.1.10",
    dac_port=1234,
    dac_channel=40,            # 你说的 channel
    dac_cfg_param=16.284,      # 你代码里第三个参数
    v_start=0.0,
    v_stop=0.5,
    v_step=0.01,
    settle_s=0.02,
    ni_physical_ch="cDAQ9185-2395476Mod3/ai2",
    out_csv="sweep_0_0p5V_ch40.csv",
):
    # Load MCP.dll
    lib = ctypes.cdll.LoadLibrary(dll_path)
    lib.makeSetProtocol.argtypes = [c_ushort, c_double, c_double]
    lib.makeSetProtocol.restype = MCPSEND

    # UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = (dac_ip, dac_port)

    # Voltage list (include endpoint)
    n_steps = int(round((v_stop - v_start) / v_step)) + 1
    voltages = [v_start + i * v_step for i in range(n_steps)]

    print("Start sweep:")
    print(f"  DAC: {dac_ip}:{dac_port}, channel={dac_channel}, cfg={dac_cfg_param}")
    print(f"  NI : {ni_physical_ch}")
    print(f"  V  : {v_start} -> {v_stop} step {v_step}")
    print(f"  CSV: {out_csv}\n")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([ "v_set_V", "ni_mean_V"])

        try:
            # (可选) 先把该通道置 0V 一次，类似你原来的初始化
            pkt0 = lib.makeSetProtocol(dac_channel, 0.0, float(dac_cfg_param))
            sock.sendto(_bytes_from_send(pkt0), addr)
            time.sleep(0.01)

            for v in voltages:
                # 1) Set DAC voltage
                pkt = lib.makeSetProtocol(dac_channel, float(v), float(dac_cfg_param))
                sock.sendto(_bytes_from_send(pkt), addr)

                # 2) Wait settle
                time.sleep(settle_s)

                # 3) Read NI mean (5 points)
                ni_mean = ni9215_read_5pt_mean(ni_physical_ch)

                # 4) Save

                writer.writerow([ float(v), float(ni_mean)])

                print(f"V_set={v:.3f} V -> NI_mean={ni_mean:.6f} V")


        finally:

            # 结束复位：把 DAC 拉回 0V（推荐）

            try:

                pkt_end = lib.makeSetProtocol(dac_channel, 0.0, float(dac_cfg_param))

                sock.sendto(_bytes_from_send(pkt_end), addr)

                time.sleep(0.01)

                print("DAC reset to 0V.")

            except Exception as e:

                print(f"WARNING: DAC reset failed: {e}")

            sock.close()

    print("\nDone.")


if __name__ == "__main__":
    sweep_0_to_0p5V_and_log(
        # 你已验证正确的 NI 通道
        ni_physical_ch="cDAQ9185-2395476Mod1/ai0",

        # 扫描参数（你可改步长、稳定等待时间）
        v_start=0.0,
        v_stop=1,
        v_step=0.005,
        settle_s=0.01,

        # 输出 CSV
        out_csv="balanced_1V_ch39_155540.csv",

        # DAC 通道与配置
        dac_channel=39,
        dac_cfg_param=16.284,
    )
