"""Statistical report generator for evaluation results."""

from scripts.eval_stats import (
    LANGUAGE_META,
    badcase_entries,
    band_distribution,
    collect_weighted_scores,
    summarize,
)


class ReportGenerator:
    def __init__(self, results: list, language: str = "en"):
        self.results = results
        self.language = language

    def print_summary(self):
        total = len(self.results)
        success = sum(1 for r in self.results if r["status"] == "success")
        empty = sum(1 for r in self.results if r["status"] == "empty")
        failed = sum(1 for r in self.results if r["status"] == "failed")
        retrieval_missing = sum(
            1 for r in self.results
            if (r.get("result") or {}).get("_retrieval_status", "available") != "available"
        )

        print()
        print("=" * 60)
        print("  评测完成情况")
        print("=" * 60)
        print(f"  评测总数:   {total}")
        print(f"  成功:        {success}")
        print(f"  回答为空:    {empty}")
        print(f"  失败:        {failed}")
        print(f"  检索缺失:    {retrieval_missing}")
        print()

        meta = LANGUAGE_META.get(self.language, LANGUAGE_META["en"])
        dim_keys = meta.dims
        score_key = meta.score_key

        dim_scores = {k: [] for k in dim_keys}
        weighted_pairs = collect_weighted_scores(self.results, self.language)
        weighted_scores = [score for _, score in weighted_pairs]

        for r in self.results:
            if r["status"] not in ("success", "empty"):
                continue
            result = r.get("result") or {}
            for dim_key in dim_keys:
                dim_data = result.get(dim_key, {})
                s = dim_data.get(score_key)
                if isinstance(s, (int, float)):
                    dim_scores[dim_key].append(int(s))
                else:
                    dim_scores[dim_key].append(0)

        print("-" * 60)
        print("  各维度统计")
        print("-" * 60)
        print(f"  {'维度':<30} {'均值':>6} {'中位数':>7} {'标准差':>7}")
        print("  " + "-" * 52)
        for dim_key in dim_keys:
            scores = [s for s in dim_scores[dim_key] if s > 0]
            if scores:
                stats = summarize(scores)
                avg = stats["mean"]
                med = stats["median"]
                std = stats["stdev"]
                print(f"  {dim_key:<30} {avg:>6.2f} {med:>7.1f} {std:>7.2f}")
        print()

        if weighted_scores:
            ws_stats = summarize(weighted_scores)
            avg_w = ws_stats["mean"]
            max_w = ws_stats["max"]
            min_w = ws_stats["min"]
            print("-" * 60)
            print("  加权均分分布")
            print("-" * 60)
            print(f"  均值: {avg_w:.2f}")
            print(f"  最高: {max_w:.2f}")
            print(f"  最低: {min_w:.2f}")
            print()

            bands = band_distribution(weighted_scores)

            print("-" * 60)
            print("  分数段分布")
            print("-" * 60)
            for band, count in bands.items():
                pct = count / len(weighted_scores) * 100 if weighted_scores else 0
                bar = "#" * int(pct / 2)
                print(f"  {band:<10} {count:>4} ({pct:>5.1f}%)  {bar}")
            print()

            badcases = badcase_entries(weighted_pairs)
            badcase_rate = len(badcases) / len(weighted_scores) * 100 if weighted_scores else 0

            print("-" * 60)
            print("  Badcase列表（加权均分 < 3.0）")
            print("-" * 60)
            if badcases:
                print(f"  Badcase率: {badcase_rate:.1f}% ({len(badcases)}/{len(weighted_scores)})")
                print()
                for idx, w in badcases:
                    item = next((r for r in self.results if r.get("index") == idx), None)
                    query_text = ""
                    if item:
                        result = item.get("result") or {}
                        if isinstance(result, dict):
                            query_text = result.get("query", "")
                    query_short = query_text[:60] if query_text else f"entry #{idx}"
                    print(f"  #{idx:>4}  {w:.2f}  {query_short}")
            else:
                print("  无Badcase。")
            print()

        print("=" * 60)