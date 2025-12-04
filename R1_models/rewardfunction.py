def _get_reward(self):
    qoe = self.aircraft.get_qoe_metrics(self.aircraft.deltaT)
    if not qoe or "throughput_req_mbps" not in qoe or "latency_req_s" not in qoe:
        return 0.0

    print(f"QoE metrics: {qoe}")

    deltaT = self.aircraft.deltaT

    # --- Throughput satisfaction ---
    throughput_req = qoe["throughput_req_mbps"]          # sum of min app throughputs (Mbps)
    allocated_MB   = qoe["allocated_bandwidth_MB"]       # MB over this timestep
    allocated_mbps = (allocated_MB * 8.0) / deltaT       # Mb / s

    if throughput_req > 0:
        throughput_satisfaction = min(allocated_mbps / throughput_req, 1.0)
    else:
        throughput_satisfaction = 1.0

    # --- Latency satisfaction ---
    total_latency_s = qoe["queuing_delay_s"] + qoe["propagation_latency_s"]
    latency_req_s   = qoe["latency_req_s"]

    if latency_req_s > 0:
        if total_latency_s <= latency_req_s:
            latency_satisfaction = 1.0
        else:
            # degrade linearly from 1 at threshold to 0 at 2×threshold
            ratio = total_latency_s / latency_req_s
            latency_satisfaction = max(0.0, 1.0 - (ratio - 1.0))
    else:
        latency_satisfaction = 1.0

    # --- Combine throughput + latency ---
    w_thr = 0.7   # throughput weight
    w_lat = 0.3   # latency weight

    reward = (
        w_thr * throughput_satisfaction +
        w_lat * latency_satisfaction
    )

    # Optional: handover penalty if you track it
    # if self.handover_happened:
    #     reward -= 0.05

    return float(reward)