import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import { Activity, BarChart3, Search, ShieldAlert, Trophy } from "lucide-react";

const API_BASE = "http://127.0.0.1:8000";

interface ShapFactor {
  feature: string;
  impact: number;
  description: string;
}

interface MatchPrediction {
  team_1: string;
  team_2: string;
  venue: string;
  competition: string;
  predicted_winner: string;
  team_1_win_prob: number;
  team_2_win_prob: number;
  confidence_level: string;
  expected_first_innings_score: number;
  pp_collapse_prob: number;
  positive_factors_t1: ShapFactor[];
  negative_factors_t1: ShapFactor[];
  uncertainty_notes: string[];
}

interface PlayerProjection {
  player_id: string;
  player_name: string;
  latest_team: string;
  bat_innings: number;
  bat_sr_10: number | null;
  proj_runs: number;
  prob_20plus: number;
  prob_30plus: number;
  bowl_innings: number;
  bowl_econ_10: number | null;
  proj_wickets: number;
  proj_economy: number;
}

interface RecentMatch {
  match_id: string;
  match_date: string;
  venue_clean: string;
  team_1: string;
  team_2: string;
  winner: string;
  innings_1_runs: number | null;
  innings_2_runs: number | null;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<"predict" | "players" | "history">("predict");
  const [competition, setCompetition] = useState<string>("ilt20");
  const [teams, setTeams] = useState<string[]>([]);
  const [venues, setVenues] = useState<string[]>([]);

  const [team1, setTeam1] = useState<string>("Dubai Capitals");
  const [team2, setTeam2] = useState<string>("Desert Vipers");
  const [venue, setVenue] = useState<string>("Dubai International Cricket Stadium");
  const [tossWinner, setTossWinner] = useState<string>("");
  const [tossDecision, setTossDecision] = useState<string>("field");

  const [prediction, setPrediction] = useState<MatchPrediction | null>(null);
  const [loadingPred, setLoadingPred] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string>("");

  const [playerQuery, setPlayerQuery] = useState<string>("Fakhar");
  const [players, setPlayers] = useState<PlayerProjection[]>([]);
  const [recentMatches, setRecentMatches] = useState<RecentMatch[]>([]);
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    async function loadOptions() {
      try {
        const [tRes, vRes, mRes] = await Promise.all([
          fetch(`${API_BASE}/api/teams?competition=${competition}`),
          fetch(`${API_BASE}/api/venues?competition=${competition}`),
          fetch(`${API_BASE}/api/matches/recent?competition=${competition}&limit=10`),
        ]);
        const tData = await tRes.json();
        const vData = await vRes.json();
        const mData = await mRes.json();

        setTeams(tData.teams || []);
        setVenues(vData.venues || []);
        setRecentMatches(mData.matches || []);

        if (tData.teams?.length >= 2) {
          setTeam1(tData.teams[0]);
          setTeam2(tData.teams[1]);
        }
        if (vData.venues?.length >= 1) {
          setVenue(vData.venues[0]);
        }
        setTossWinner("");
      } catch {
        setErrorMsg("Cannot reach FastAPI backend at http://127.0.0.1:8000. Is Uvicorn running?");
      }
    }
    loadOptions();
  }, [competition]);

  useEffect(() => {
    fetch(`${API_BASE}/api/metrics`)
      .then((r) => r.json())
      .then(setMetrics)
      .catch(() => {});
  }, []);

  async function handlePredict(e: React.FormEvent) {
    e.preventDefault();
    if (team1 === team2) {
      setErrorMsg("Please select two different teams.");
      return;
    }
    setErrorMsg("");
    setLoadingPred(true);
    try {
      const res = await fetch(`${API_BASE}/api/predict/match`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          team_1: team1,
          team_2: team2,
          venue,
          competition,
          toss_winner: tossWinner || null,
          toss_decision: tossWinner ? tossDecision : null,
        }),
      });
      if (!res.ok) throw new Error("Prediction request failed");
      const data: MatchPrediction = await res.json();
      setPrediction(data);
    } catch {
      setErrorMsg("Failed to fetch prediction from API.");
    } finally {
      setLoadingPred(false);
    }
  }

  async function handlePlayerSearch(e: React.FormEvent) {
    e.preventDefault();
    if (playerQuery.trim().length < 2) return;
    const res = await fetch(`${API_BASE}/api/players/search?q=${encodeURIComponent(playerQuery)}&limit=12`);
    const data = await res.json();
    setPlayers(data.players || []);
  }

  const shapChartData = prediction
    ? [...prediction.positive_factors_t1, ...prediction.negative_factors_t1].sort(
        (a, b) => a.impact - b.impact
      )
    : [];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/70 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-4 px-6 py-4">
          <div className="flex items-center gap-3">
            <Activity className="h-7 w-7 text-emerald-400" />
            <div>
              <h1 className="text-xl font-bold tracking-tight">CricPredict AI</h1>
              <p className="text-xs text-slate-400">Explainable T20 & ILT20 Analytics Platform</p>
            </div>
          </div>

          <nav className="flex gap-2">
            {[
              { id: "predict", label: "Match Predictor", icon: Trophy },
              { id: "players", label: "Player Projections", icon: Search },
              { id: "history", label: "Matches & Metrics", icon: BarChart3 },
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id as typeof activeTab)}
                className={`flex items-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition ${
                  activeTab === id
                    ? "bg-emerald-500 text-slate-950"
                    : "bg-slate-800 text-slate-300 hover:bg-slate-700"
                }`}
              >
                <Icon className="h-4 w-4" />
                {label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-6 py-8">
        {errorMsg && (
          <div className="mb-6 rounded-lg border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-300">
            {errorMsg}
          </div>
        )}

        {activeTab === "predict" && (
          <div className="grid gap-6 lg:grid-cols-12">
            <form
              onSubmit={handlePredict}
              className="space-y-4 rounded-xl border border-slate-800 bg-slate-900 p-6 lg:col-span-4"
            >
              <h2 className="text-lg font-semibold">Match Setup</h2>

              <div>
                <label className="mb-1 block text-xs text-slate-400">Competition</label>
                <select
                  value={competition}
                  onChange={(e) => setCompetition(e.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
                >
                  <option value="ilt20">ILT20 (International League T20)</option>
                  <option value="bbl">BBL (Big Bash League)</option>
                  <option value="t20i">T20 Internationals</option>
                </select>
              </div>

              <div>
                <label className="mb-1 block text-xs text-slate-400">Team 1</label>
                <select
                  value={team1}
                  onChange={(e) => setTeam1(e.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
                >
                  {teams.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="mb-1 block text-xs text-slate-400">Team 2</label>
                <select
                  value={team2}
                  onChange={(e) => setTeam2(e.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
                >
                  {teams.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="mb-1 block text-xs text-slate-400">Venue</label>
                <select
                  value={venue}
                  onChange={(e) => setVenue(e.target.value)}
                  className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
                >
                  {venues.map((v) => (
                    <option key={v} value={v}>{v}</option>
                  ))}
                </select>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="mb-1 block text-xs text-slate-400">Toss Winner (Optional)</label>
                  <select
                    value={tossWinner}
                    onChange={(e) => setTossWinner(e.target.value)}
                    className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm"
                  >
                    <option value="">Pre-Toss</option>
                    <option value={team1}>{team1}</option>
                    <option value={team2}>{team2}</option>
                  </select>
                </div>
                <div>
                  <label className="mb-1 block text-xs text-slate-400">Decision</label>
                  <select
                    value={tossDecision}
                    onChange={(e) => setTossDecision(e.target.value)}
                    disabled={!tossWinner}
                    className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm disabled:opacity-40"
                  >
                    <option value="field">Field</option>
                    <option value="bat">Bat</option>
                  </select>
                </div>
              </div>

              <button
                type="submit"
                disabled={loadingPred}
                className="w-full rounded-lg bg-emerald-500 py-2.5 text-sm font-semibold text-slate-950 transition hover:bg-emerald-400 disabled:opacity-50"
              >
                {loadingPred ? "Running LightGBM + SHAP..." : "Predict Match Outcome"}
              </button>
            </form>

            <div className="space-y-6 lg:col-span-8">
              {!prediction ? (
                <div className="flex h-full min-h-64 items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-900/40 p-8 text-center text-sm text-slate-400">
                  Select two teams and a venue, then click <strong className="mx-1 text-slate-200">Predict Match Outcome</strong> to view probabilities and SHAP explanations.
                </div>
              ) : (
                <>
                  <div className="rounded-xl border border-slate-800 bg-slate-900 p-6">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div>
                        <p className="text-xs uppercase tracking-wider text-emerald-400">
                          {prediction.competition.toUpperCase()} • {prediction.venue}
                        </p>
                        <h3 className="mt-1 text-2xl font-bold">
                          Predicted Winner: {prediction.predicted_winner}
                        </h3>
                      </div>
                      <span className="rounded-full border border-slate-700 bg-slate-800 px-3 py-1 text-xs font-medium text-slate-200">
                        Confidence: {prediction.confidence_level}
                      </span>
                    </div>

                    <div className="mt-5">
                      <div className="mb-2 flex justify-between text-sm font-semibold">
                        <span>{prediction.team_1}: {(prediction.team_1_win_prob * 100).toFixed(1)}%</span>
                        <span>{prediction.team_2}: {(prediction.team_2_win_prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex h-4 overflow-hidden rounded-full bg-slate-800">
                        <div
                          className="bg-emerald-500 transition-all"
                          style={{ width: `${prediction.team_1_win_prob * 100}%` }}
                        />
                        <div
                          className="bg-sky-500 transition-all"
                          style={{ width: `${prediction.team_2_win_prob * 100}%` }}
                        />
                      </div>
                    </div>

                    <div className="mt-6 grid gap-4 sm:grid-cols-2">
                      <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-4">
                        <p className="text-xs text-slate-400">Expected 1st Innings Score</p>
                        <p className="mt-1 text-2xl font-bold text-slate-100">
                          {prediction.expected_first_innings_score} runs
                        </p>
                      </div>
                      <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-4">
                        <p className="text-xs text-slate-400">Powerplay Collapse Risk (3+ Wkts)</p>
                        <p className="mt-1 text-2xl font-bold text-amber-400">
                          {(prediction.pp_collapse_prob * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>

                    {prediction.uncertainty_notes.length > 0 && (
                      <div className="mt-4 flex items-start gap-2 rounded-lg border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200">
                        <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0" />
                        <ul className="list-disc space-y-1 pl-4">
                          {prediction.uncertainty_notes.map((n, i) => (
                            <li key={i}>{n}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>

                  <div className="rounded-xl border border-slate-800 bg-slate-900 p-6">
                    <h4 className="mb-1 text-base font-semibold">
                      SHAP Feature Contributions (Impact on {prediction.team_1})
                    </h4>
                    <p className="mb-4 text-xs text-slate-400">
                      Green bars push {prediction.team_1}&apos;s win probability up; red bars push it down.
                    </p>
                    <div className="h-64 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={shapChartData} layout="vertical" margin={{ left: 20, right: 20 }}>
                          <XAxis type="number" stroke="#94a3b8" fontSize={12} />
                          <YAxis type="category" dataKey="description" width={210} stroke="#cbd5e1" fontSize={11} />
                          <Tooltip
                            contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", color: "#f8fafc" }}
                          />
                          <ReferenceLine x={0} stroke="#64748b" />
                          <Bar dataKey="impact" radius={[4, 4, 4, 4]}>
                            {shapChartData.map((entry, idx) => (
                              <Cell key={idx} fill={entry.impact >= 0 ? "#10b981" : "#f43f5e"} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {activeTab === "players" && (
          <div className="space-y-6">
            <form onSubmit={handlePlayerSearch} className="flex gap-3">
              <input
                type="text"
                value={playerQuery}
                onChange={(e) => setPlayerQuery(e.target.value)}
                placeholder="Search player name (e.g. Fakhar, du Plessis, Mustafizur)..."
                className="flex-1 rounded-lg border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm"
              />
              <button
                type="submit"
                className="rounded-lg bg-emerald-500 px-5 py-2.5 text-sm font-semibold text-slate-950 hover:bg-emerald-400"
              >
                Search
              </button>
            </form>

            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
              {players.map((p) => (
                <div key={p.player_id} className="rounded-xl border border-slate-800 bg-slate-900 p-5">
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="font-semibold text-slate-100">{p.player_name}</h3>
                      <p className="text-xs text-slate-400">{p.latest_team}</p>
                    </div>
                    <span className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-300">
                      {p.bat_innings} bat / {p.bowl_innings} bowl
                    </span>
                  </div>
                  <div className="mt-4 grid grid-cols-2 gap-3 text-xs">
                    <div className="rounded-lg bg-slate-950/60 p-2.5">
                      <p className="text-slate-400">Proj. Runs</p>
                      <p className="text-base font-bold text-emerald-400">{p.proj_runs}</p>
                    </div>
                    <div className="rounded-lg bg-slate-950/60 p-2.5">
                      <p className="text-slate-400">20+ / 30+ Prob</p>
                      <p className="text-base font-bold text-sky-400">
                        {(p.prob_20plus * 100).toFixed(0)}% / {(p.prob_30plus * 100).toFixed(0)}%
                      </p>
                    </div>
                    <div className="rounded-lg bg-slate-950/60 p-2.5">
                      <p className="text-slate-400">Proj. Wickets</p>
                      <p className="text-base font-bold text-amber-400">{p.proj_wickets}</p>
                    </div>
                    <div className="rounded-lg bg-slate-950/60 p-2.5">
                      <p className="text-slate-400">Proj. Economy</p>
                      <p className="text-base font-bold text-slate-200">{p.proj_economy}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === "history" && (
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="rounded-xl border border-slate-800 bg-slate-900 p-6">
              <h2 className="mb-4 text-lg font-semibold">Recent {competition.toUpperCase()} Matches</h2>
              <div className="space-y-3">
                {recentMatches.map((m) => (
                  <div key={m.match_id} className="rounded-lg border border-slate-800 bg-slate-950/60 p-3.5 text-sm">
                    <div className="flex justify-between text-xs text-slate-400">
                      <span>{m.match_date}</span>
                      <span>{m.venue_clean}</span>
                    </div>
                    <p className="mt-1 font-medium">
                      {m.team_1} vs {m.team_2}
                    </p>
                    <p className="mt-1 text-xs text-emerald-400">
                      Winner: {m.winner} ({m.innings_1_runs ?? "-"} vs {m.innings_2_runs ?? "-"})
                    </p>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-xl border border-slate-800 bg-slate-900 p-6">
              <h2 className="mb-4 text-lg font-semibold">Chronological Model Evaluation</h2>
              <pre className="max-h-[480px] overflow-auto rounded-lg bg-slate-950 p-4 text-xs text-slate-300">
                {JSON.stringify(metrics, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}