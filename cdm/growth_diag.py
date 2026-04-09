def compute_raw_expert_request(novelty, threshold, novelty_step):
    if novelty <= threshold:
        return 0
    step = max(float(novelty_step), 1e-8)
    return 1 + int((novelty - threshold) / step)


def build_growth_plan_and_diagnostics(layer_novelty, layer_adapters, max_experts_per_layer,
                                      layer_growth_topk, novelty_step, layer_threshold_fn,
                                      compute_new_experts_fn):
    ranked = []
    diagnostics = {}
    for key, novelty in layer_novelty.items():
        threshold = layer_threshold_fn(key)
        adapter = layer_adapters[key]
        current = adapter.num_experts
        raw_request = compute_raw_expert_request(novelty, threshold, novelty_step)
        room = float("inf") if max_experts_per_layer is None else max(max_experts_per_layer - current, 0)
        at_cap = (max_experts_per_layer is not None and room == 0)
        diagnostics[key] = {
            "current_experts": current,
            "max_experts": max_experts_per_layer,
            "novelty": novelty,
            "threshold": threshold,
            "raw_request": raw_request,
            "room": room,
            "actual_request": 0,
            "dropped_by_topk": False,
            "at_cap": at_cap,
            "stuck_high_novelty_room0": (novelty > threshold and at_cap),
        }
        if novelty > threshold:
            ranked.append((key, novelty, threshold))

    ranked.sort(key=lambda x: x[1], reverse=True)
    if layer_growth_topk > 0:
        dropped_keys = {key for key, _, _ in ranked[layer_growth_topk:]}
        ranked = ranked[:layer_growth_topk]
        for key in dropped_keys:
            diagnostics[key]["dropped_by_topk"] = True

    growth_plan = {}
    for key, novelty, threshold in ranked:
        current = layer_adapters[key].num_experts
        num_new_experts = compute_new_experts_fn(novelty, current_experts=current, threshold=threshold)
        if num_new_experts > 0:
            growth_plan[key] = num_new_experts
        diagnostics[key]["actual_request"] = int(max(num_new_experts, 0))

    return growth_plan, diagnostics


def log_growth_diagnostics(logger, diagnostics):
    if len(diagnostics) == 0:
        logger.info("[WarmupGrowDiag] no layers had valid novelty statistics.")
        return
    for key in sorted(diagnostics.keys()):
        d = diagnostics[key]
        max_experts = "None" if d["max_experts"] is None else str(d["max_experts"])
        logger.info(
            f"[WarmupGrowDiag] layer={key} "
            f"experts={d['current_experts']}/{max_experts} "
            f"novelty={d['novelty']:.4f} threshold={d['threshold']:.4f} "
            f"raw_request={d['raw_request']} actual_request={d['actual_request']} "
            f"dropped_by_topk={d['dropped_by_topk']} at_cap={d['at_cap']} "
            f"stuck_high_novelty_room0={d['stuck_high_novelty_room0']}"
        )
