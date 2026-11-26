import { useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

const API_BASE = "http://127.0.0.1:8000";

interface LiveResponse {
  ml_chase_win_prob: number;
  monte_carlo_win_prob: number;
  current_run_rate: number;
  required_run_rate: number;
  projected_score_p10: number;
  projected_score_p50: number;
  projected_score_p90: number;
}

interface TimelinePoint {
  ballLabel: string;
  mlProb: number;
  mcProb: number;
  score: string;
}

export default function LiveMatchCentre() {
  const [target, setTarget] = useState(175);
  const [score, setScore] = useState(110);
  const [balls, setBalls] = useState(78);
  const [wickets, setWickets] = useState(3);
  const [competition, setCompetition] = useState("ilt20");
  const [live, setLive] = useState<LiveResponse | null>(null);
  const [timeline, setTimeline] = useState<TimelinePoint[]>([]);

  async function runLiveUpdate(
    nextScore: number,
    nextBalls: number,
    nextWickets: number,
    recentRuns: number,
    recentWkts: number
  ) {
    const res = await fetch(`${API_BASE}/api/predict/live`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        target,
        current_score: nextScore,
        balls_bowled: nextBalls,
        wickets_lost: nextWickets,
        last_18_runs: recentRuns,
        last_18_wkts: recentWkts,
        competition,
      }),
    });
    const data: LiveResponse = await res.json();
    setLive(data);

    const oversLabel = `${Math.floor(nextBalls / 6)}.${nextBalls % 6}`;
    setTimeline((prev) => [
      ...prev.slice(-24),
      {
        ballLabel: oversLabel,
        mlProb: Number((data.ml_chase_win_prob * 100).toFixed(1)),
        mcProb: Number((data.monte_carlo_win_prob * 100).toFixed(1)),
        score: `${nextScore}/${nextWickets}`,
      },
    ]);
  }

  function simulateBall(runs: number, isWicket = false) {
    if (balls >= 120 || wickets >= 10 || score >= target) return;
    const nextScore = score + runs;
    const nextBalls = balls + 1;
    const nextWkts = isWicket ? wickets + 1 : wickets;
    setScore(nextScore);
    setBalls(nextBalls);
    setWickets(nextWkts);
    runLiveUpdate(nextScore, nextBalls, nextWkts, 24 + runs, isWicket ? 1 : 0);
  }

  return (
    <div className="grid gap-6 lg:grid-cols-12">
      <div className="space-y-4 rounded-xl border border-slate-800 bg-slate-900 p-6 lg:col-span-4">
        <h2 className="text-lg font-semibold">Live Chase State</h2>

        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <label className="mb-1 block text-xs text-slate-400">Target</label>
            <input
              type="number"
              value={target}
              onChange={(e) => setTarget(Number(e.target.value))}
              className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-slate-400">Competition</label>
            <select
              value={competition}
              onChange={(e) => setCompetition(e.target.value)}
              className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2"
            >
              <option value="ilt20">ILT20</option>
              <option value="bbl">BBL</option>
              <option value="t20i">T20I</option>
            </select>
          </div>
          <div>
            <label className="mb-1 block text-xs text-slate-400">Score</label>
            <input
              type="number"
              value={score}
              onChange={(e) => setScore(Number(e.target.value))}
              className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2"
            />
          </div>
          <div>
            <label className="mb-1 block text-xs text-slate-400">Wickets Lost</label>
            <input
              type="number"
              min={0}
              max={10}
              value={wickets}
              onChange={(e) => setWickets(Number(e.target.value))}
              className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2"
            />
          </div>
        </div>

        <div>
          <label className="mb-1 block text-xs text-slate-400">
            Balls Bowled: {balls} ({Math.floor(balls / 6)}.{balls % 6} overs)
          </label>
          <input
            type="range"
            min={0}
            max={119}
            value={balls}
            onChange={(e) => setBalls(Number(e.target.value))}
            className="w-full accent-emerald-500"
          />
        </div>

        <button
          onClick={() => runLiveUpdate(score, balls, wickets, 24, 0)}
          className="w-full rounded-lg bg-emerald-500 py-2.5 text-sm font-semibold text-slate-950 hover:bg-emerald-400"
        >
          Evaluate Current State
        </button>

        <div className="border-t border-slate-800 pt-4">
          <p className="mb-2 text-xs text-slate-400">Simulate Next Ball</p>
          <div className="grid grid-cols-5 gap-2 text-xs font-semibold">
            <button onClick={() => simulateBall(0)} className="rounded bg-slate-800 py-2 hover:bg-slate-700">Dot</button>
            <button onClick={() => simulateBall(1)} className="rounded bg-slate-800 py-2 hover:bg-slate-700">+1</button>
            <button onClick={() => simulateBall(4)} className="rounded bg-sky-600 py-2 hover:bg-sky-500">+4</button>
            <button onClick={() => simulateBall(6)} className="rounded bg-emerald-600 py-2 hover:bg-emerald-500">+6</button>
            <button onClick={() => simulateBall(0, true)} className="rounded bg-rose-600 py-2 hover:bg-rose-500">WKT</button>
          </div>
        </div>
      </div>

      <div className="space-y-6 lg:col-span-8">
        {live && (
          <div className="grid gap-4 sm:grid-cols-4">
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
              <p className="text-xs text-slate-400">ML Chase Win %</p>
              <p className="mt-1 text-2xl font-bold text-emerald-400">
                {(live.ml_chase_win_prob * 100).toFixed(1)}%
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
              <p className="text-xs text-slate-400">Monte Carlo Win %</p>
              <p className="mt-1 text-2xl font-bold text-sky-400">
                {(live.monte_carlo_win_prob * 100).toFixed(1)}%
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
              <p className="text-xs text-slate-400">CRR / RRR</p>
              <p className="mt-1 text-xl font-bold text-slate-100">
                {live.current_run_rate} / {live.required_run_rate}
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-4">
              <p className="text-xs text-slate-400">Proj. Score (P10–P90)</p>
              <p className="mt-1 text-xl font-bold text-amber-400">
                {live.projected_score_p50} ({live.projected_score_p10}–{live.projected_score_p90})
              </p>
            </div>
          </div>
        )}

        <div className="rounded-xl border border-slate-800 bg-slate-900 p-6">
          <h3 className="mb-4 text-base font-semibold">Live Win Probability Timeline</h3>
          {timeline.length === 0 ? (
            <p className="py-12 text-center text-sm text-slate-400">
              Click <strong>Evaluate Current State</strong> or simulate balls to plot the live probability curve.
            </p>
          ) : (
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={timeline}>
                  <XAxis dataKey="ballLabel" stroke="#94a3b8" fontSize={12} />
                  <YAxis domain={[0, 100]} stroke="#94a3b8" fontSize={12} />
                  <Tooltip contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155" }} />
                  <ReferenceLine y={50} stroke="#64748b" strokeDasharray="3 3" />
                  <Area type="monotone" dataKey="mlProb" name="ML Win %" stroke="#10b981" fill="#10b981" fillOpacity={0.2} />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}