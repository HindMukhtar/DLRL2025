import os

import pandas as pd
import simpy

from LEOEnvironmentRL import initialize, load_route_from_csv


def main():
    base_dir = os.path.dirname(__file__)
    route_path = os.path.join(base_dir, "route_5s_interpolated.csv")
    scenarios = ["no_scenario", "demand_aware", "snr_congested", "peak_hour", "large_aircraft"]
    route, _ = load_route_from_csv(route_path, skip_rows=0)
    output_path = os.path.join(base_dir, "beam_availability_all_scenarios.txt")

    envs = {}
    for scenario in scenarios:
        scenario_arg = None if scenario == "no_scenario" else scenario
        env = simpy.Environment()
        earth = initialize(env, "OneWeb", route, scenario=scenario_arg)
        envs[scenario] = {
            "env": env,
            "earth": earth,
            "aircraft": earth.aircraft[0],
        }

    header_cols = [
        "step",
        "time_s",
        "beam_id",
        "sat_id",
        "snr_db",
        "capacity",
    ]
    header_cols.extend([f"load_{scenario}" for scenario in scenarios])

    with open(output_path, "w") as f:
        f.write(",".join(header_cols) + "\n")

        for step in range(len(route) - 1):
            scenario_maps = {}
            sim_time = None

            for scenario in scenarios:
                env = envs[scenario]["env"]
                earth = envs[scenario]["earth"]
                aircraft = envs[scenario]["aircraft"]

                delta_t = aircraft.move_and_connect_aircraft(earth.LEO)
                earth.deltaT = delta_t
                candidates = aircraft.scan_nearby_fast(earth.LEO)
                scenario_maps[scenario] = {
                    c["beam"].id: c for c in candidates
                }
                if sim_time is None:
                    sim_time = env.now

                earth.advance_constellation(earth.deltaT, env.now)
                env.run(until=env.now + earth.deltaT)

            all_beam_ids = set()
            for scenario in scenarios:
                all_beam_ids.update(scenario_maps[scenario].keys())

            for beam_id in sorted(all_beam_ids):
                base_candidate = scenario_maps["no_scenario"].get(beam_id)
                sat_id = base_candidate["sat"].ID if base_candidate else ""
                snr_db = f"{base_candidate['snr']:.3f}" if base_candidate else ""
                capacity = f"{base_candidate['capacity']:.6f}" if base_candidate else ""

                row = [
                    str(step),
                    f"{sim_time:.2f}",
                    beam_id,
                    sat_id,
                    snr_db,
                    capacity,
                ]
                for scenario in scenarios:
                    candidate = scenario_maps[scenario].get(beam_id)
                    if candidate:
                        row.append(f"{candidate['load']:.3f}")
                    else:
                        row.append("")

                f.write(",".join(row) + "\n")

    print(f"Wrote beam availability to {output_path}")


if __name__ == "__main__":
    main()
