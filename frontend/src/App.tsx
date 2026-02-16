import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

type Action = 'U' | 'D' | 'L' | 'R'

type GridSpec = {
    H: number
    W: number
    start: [number, number]
    goal: [number, number]
    walls: [number, number][]
}

type EvalPoint = {
    episode: number
    avg_return: number
}

type Snapshot = {
    grid: GridSpec
    episode: number
    epsilon: number
    Q: Record<string, Record<Action, number>>
    policy: Record<string, Action | 'G' | null>
    reward_history: number[]
    eval_history: EvalPoint[]
}

type SeriesPoint = {
    episode: number
    value: number
}

type ChartResult = {
    trainingPath: string
    yTicks: Array<{ y: number; label: string }>
    xTicks: Array<{ x: number; label: string; anchor: 'start' | 'middle' | 'end' }>
}

const API_BASE = import.meta.env.VITE_API_BASE ?? 'http://localhost:8000'
const EVAL_EVERY = 50
const N_EVAL = 20

const ARROW_BY_ACTION: Record<Action, string> = {
    U: '↑',
    D: '↓',
    L: '←',
    R: '→',
}

const CHART_WIDTH = 560
const CHART_HEIGHT = 280
const CHART_MARGIN = { top: 10, right: 20, bottom: 60, left: 58 }
const CHART_INNER_WIDTH = CHART_WIDTH - CHART_MARGIN.left - CHART_MARGIN.right
const CHART_INNER_HEIGHT = CHART_HEIGHT - CHART_MARGIN.top - CHART_MARGIN.bottom
const LEARNING_CURVE_Y_MIN = -55
const LEARNING_CURVE_Y_MAX = 10
const LEARNING_CURVE_Y_TICK_STEP = 5

function keyOf(r: number, c: number) {
    return `${r},${c}`
}

function greedyPath(snapshot: Snapshot): string[] {
    const path: string[] = []
    const seen = new Set<string>()
    const [goalR, goalC] = snapshot.grid.goal
    let [r, c] = snapshot.grid.start

    for (let i = 0; i < 30; i += 1) {
        const key = keyOf(r, c)
        path.push(key)
        if (r === goalR && c === goalC) break
        if (seen.has(key)) break
        seen.add(key)

        const action = snapshot.policy[key]
        if (!action || action === 'G') break

        let nr = r
        let nc = c
        if (action === 'U') nr -= 1
        if (action === 'D') nr += 1
        if (action === 'L') nc -= 1
        if (action === 'R') nc += 1

        const blocked = snapshot.grid.walls.some(([wr, wc]) => wr === nr && wc === nc)
        const outOfBounds = nr < 0 || nr >= snapshot.grid.H || nc < 0 || nc >= snapshot.grid.W
        if (blocked || outOfBounds) break

        r = nr
        c = nc
    }

    return path
}

function computeRollingAverage(data: number[], window = 50): number[] {
    const rollingAvg: number[] = []
    let runningSum = 0

    for (let i = 0; i < data.length; i += 1) {
        runningSum += data[i]
        if (i >= window) runningSum -= data[i - window]
        const count = Math.min(i + 1, window)
        rollingAvg.push(runningSum / count)
    }

    return rollingAvg
}

function formatEpisodeLabel(value: number): string {
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`
    if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`
    return `${value}`
}

function downsampleByBucket(series: number[], maxPoints: number): SeriesPoint[] {
    if (series.length <= maxPoints) return series.map((value, idx) => ({ episode: idx + 1, value }))

    const sampled: SeriesPoint[] = []
    const bucket = series.length / maxPoints

    for (let i = 0; i < maxPoints; i += 1) {
        const start = Math.floor(i * bucket)
        const endExclusive = Math.min(series.length, Math.floor((i + 1) * bucket))
        const end = endExclusive > start ? endExclusive : start + 1

        let sum = 0
        for (let j = start; j < end; j += 1) sum += series[j]
        sampled.push({ episode: end, value: sum / (end - start) })
    }

    return sampled
}

function pointsToPath(points: SeriesPoint[], maxEpisode: number, yMin: number, yRange: number): string {
    if (points.length < 2) return ''
    return points
        .map((point) => {
            const x = CHART_MARGIN.left + ((point.episode - 1) / (maxEpisode - 1)) * CHART_INNER_WIDTH
            const clampedValue = Math.max(LEARNING_CURVE_Y_MIN, Math.min(LEARNING_CURVE_Y_MAX, point.value))
            const y = CHART_MARGIN.top + (1 - (clampedValue - yMin) / yRange) * CHART_INNER_HEIGHT
            return `${x},${y}`
        })
        .join(' ')
}

function buildChart(rewardHistory: number[]): ChartResult | null {
    const trainingPoints = downsampleByBucket(computeRollingAverage(rewardHistory, 50), 600)
    if (trainingPoints.length < 2) return null
    const yMin = LEARNING_CURVE_Y_MIN
    const yMax = LEARNING_CURVE_Y_MAX
    const yRange = yMax - yMin

    const maxEpisode = Math.max(
        2,
        rewardHistory.length,
    )

    const yTicks: Array<{ y: number; label: string }> = []
    for (let value = yMax; value >= yMin; value -= LEARNING_CURVE_Y_TICK_STEP) {
        const frac = (yMax - value) / yRange
        const y = CHART_MARGIN.top + frac * CHART_INNER_HEIGHT
        yTicks.push({ y, label: `${value}` })
    }

    const xTicks = Array.from({ length: 6 }, (_, i) => {
        const frac = i / 5
        const episodeValue = Math.round(frac * maxEpisode)
        const x = CHART_MARGIN.left + frac * CHART_INNER_WIDTH
        const anchor: 'start' | 'middle' | 'end' = i === 0 ? 'start' : i === 5 ? 'end' : 'middle'
        return { x, label: formatEpisodeLabel(episodeValue), anchor }
    })

    return {
        trainingPath: pointsToPath(trainingPoints, maxEpisode, yMin, yRange),
        yTicks,
        xTicks,
    }
}

async function fetchSnapshot(path: string, init?: RequestInit): Promise<Snapshot> {
    const response = await fetch(`${API_BASE}${path}`, init)
    if (!response.ok) throw new Error(`Request failed: ${response.status}`)
    return (await response.json()) as Snapshot
}

function App() {
    const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
    const [episodesPerTrain, setEpisodesPerTrain] = useState(100)
    const [alpha, setAlpha] = useState(0.1)
    const [isLoading, setIsLoading] = useState(false)
    const [autoRun, setAutoRun] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const trainInFlightRef = useRef(false)

    const loadState = async () => {
        setIsLoading(true)
        setError(null)
        try {
            setSnapshot(await fetchSnapshot('/state'))
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error')
        } finally {
            setIsLoading(false)
        }
    }

    const train = async (n: number) => {
        if (trainInFlightRef.current) return
        trainInFlightRef.current = true
        setIsLoading(true)
        setError(null)
        try {
            const safeN = Math.max(1, Math.min(5000, Number.isFinite(n) ? n : 1))
            const query = `n=${safeN}&alpha=${alpha}&eval_every=${EVAL_EVERY}&n_eval=${N_EVAL}`
            setSnapshot(await fetchSnapshot(`/train?${query}`, { method: 'POST' }))
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error')
            setAutoRun(false)
        } finally {
            trainInFlightRef.current = false
            setIsLoading(false)
        }
    }

    const reset = async () => {
        setIsLoading(true)
        setError(null)
        try {
            setSnapshot(await fetchSnapshot('/reset', { method: 'POST' }))
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Unknown error')
        } finally {
            setIsLoading(false)
        }
    }

    useEffect(() => {
        void loadState()
    }, [])

    useEffect(() => {
        if (!autoRun || isLoading) return
        const timer = window.setTimeout(() => {
            void train(Math.max(1, Math.min(5000, episodesPerTrain)))
        }, 300)
        return () => window.clearTimeout(timer)
    }, [autoRun, isLoading, episodesPerTrain, alpha])

    const recentAverage = useMemo(() => {
        if (!snapshot || snapshot.reward_history.length === 0) return 0
        const window = snapshot.reward_history.slice(-100)
        return window.reduce((acc, value) => acc + value, 0) / window.length
    }, [snapshot])

    const latestEval = useMemo(() => {
        if (!snapshot || snapshot.eval_history.length === 0) return null
        return snapshot.eval_history[snapshot.eval_history.length - 1]
    }, [snapshot])

    const path = useMemo(() => (snapshot ? greedyPath(snapshot) : []), [snapshot])
    const pathSet = useMemo(() => new Set(path), [path])

    const chartData = useMemo(() => {
        if (!snapshot) return null
        return buildChart(snapshot.reward_history)
    }, [snapshot])

    return (
        <div className="app-shell">
            <header className="hero">
                <h1>Monte Carlo Control Lab</h1>
                <p>Train an epsilon-greedy agent and inspect policy, Q-values, and reward trends per episode.</p>
            </header>

            <section className="toolbar">
                <label>
                    Episodes / train call
                    <input
                        type="number"
                        min={1}
                        max={5000}
                        value={episodesPerTrain}
                        onChange={(e) => setEpisodesPerTrain(Number(e.target.value) || 1)}
                    />
                </label>

                <label>
                    Alpha
                    <input
                        type="number"
                        min={0.01}
                        max={1}
                        step={0.01}
                        value={alpha}
                        onChange={(e) => setAlpha(Number(e.target.value) || 0.1)}
                    />
                </label>

                <button type="button" onClick={() => void train(Math.max(1, Math.min(5000, episodesPerTrain)))} disabled={isLoading}>
                    Train Batch
                </button>
                <button type="button" className={autoRun ? 'active' : ''} onClick={() => setAutoRun((v) => !v)} disabled={isLoading && !autoRun}>
                    {autoRun ? 'Stop Auto Train' : 'Start Auto Train'}
                </button>
                <button type="button" onClick={() => void reset()} disabled={isLoading}>
                    Reset
                </button>
            </section>

            {error ? <p className="error">Backend error: {error}</p> : null}

            {!snapshot ? (
                <p className="loading">Loading snapshot...</p>
            ) : (
                <main className="dashboard">
                    <section className="stats">
                        <article>
                            <span>Episode</span>
                            <strong>{snapshot.episode}</strong>
                        </article>
                        <article>
                            <span>Epsilon</span>
                            <strong>{snapshot.epsilon.toFixed(3)}</strong>
                        </article>
                        <article>
                            <span>Avg Return (last 100)</span>
                            <strong>{recentAverage.toFixed(2)}</strong>
                        </article>
                        <article>
                            <span>Stored Returns</span>
                            <strong>{snapshot.reward_history.length}</strong>
                        </article>
                        <article>
                            <span>Latest Greedy Eval</span>
                            <strong>{latestEval ? latestEval.avg_return.toFixed(2) : 'N/A'}</strong>
                        </article>
                    </section>

                    <section className="panel grid-panel">
                        <h2>Greedy Policy Grid</h2>
                        <div
                            className="grid"
                            style={{
                                gridTemplateColumns: `repeat(${snapshot.grid.W}, minmax(70px, 1fr))`,
                            }}
                        >
                            {Array.from({ length: snapshot.grid.H * snapshot.grid.W }, (_, idx) => {
                                const r = Math.floor(idx / snapshot.grid.W)
                                const c = idx % snapshot.grid.W
                                const key = keyOf(r, c)
                                const isWall = snapshot.grid.walls.some(([wr, wc]) => wr === r && wc === c)
                                const isGoal = snapshot.grid.goal[0] === r && snapshot.grid.goal[1] === c
                                const isStart = snapshot.grid.start[0] === r && snapshot.grid.start[1] === c
                                const policy = snapshot.policy[key]
                                const marker = path.indexOf(key)

                                return (
                                    <div
                                        key={key}
                                        className={`cell${isWall ? ' wall' : ''}${isGoal ? ' goal' : ''}${isStart ? ' start' : ''}${pathSet.has(key) ? ' path' : ''}`}
                                    >
                                        <div className="cell-head">{key}</div>
                                        <div className="cell-main">
                                            {isWall ? '■' : isGoal ? 'G' : policy && policy !== 'G' ? ARROW_BY_ACTION[policy] : '·'}
                                        </div>
                                        {marker >= 0 ? <div className="cell-step">{marker}</div> : null}
                                    </div>
                                )
                            })}
                        </div>
                        <p className="caption">Highlighted cells show one greedy rollout path from start based on current policy.</p>
                    </section>

                    <section className="panel q-panel">
                        <h2>Q Function</h2>
                        <div className="q-grid" style={{ gridTemplateColumns: `repeat(${snapshot.grid.W}, minmax(120px, 1fr))` }}>
                            {Array.from({ length: snapshot.grid.H * snapshot.grid.W }, (_, idx) => {
                                const r = Math.floor(idx / snapshot.grid.W)
                                const c = idx % snapshot.grid.W
                                const key = keyOf(r, c)
                                const isWall = snapshot.grid.walls.some(([wr, wc]) => wr === r && wc === c)
                                const isGoal = snapshot.grid.goal[0] === r && snapshot.grid.goal[1] === c
                                const q = snapshot.Q[key]

                                return (
                                    <div key={key} className={`q-cell${isWall ? ' wall' : ''}${isGoal ? ' goal' : ''}`}>
                                        <div className="q-key">{key}</div>
                                        {isWall ? (
                                            <div className="q-special">WALL</div>
                                        ) : isGoal ? (
                                            <div className="q-special">GOAL</div>
                                        ) : (
                                            <>
                                                <div>U: {q.U.toFixed(2)}</div>
                                                <div>D: {q.D.toFixed(2)}</div>
                                                <div>L: {q.L.toFixed(2)}</div>
                                                <div>R: {q.R.toFixed(2)}</div>
                                            </>
                                        )}
                                    </div>
                                )
                            })}
                        </div>
                    </section>

                    <section className="panel chart-panel">
                        <h2>Learning Curve</h2>
                        <svg viewBox="0 0 560 280" role="img" aria-label="Training and greedy evaluation returns across episodes">
                            <rect
                                x={CHART_MARGIN.left}
                                y={CHART_MARGIN.top}
                                width={CHART_INNER_WIDTH}
                                height={CHART_INNER_HEIGHT}
                                className="chart-frame"
                            />
                            {chartData ? (
                                <>
                                    {chartData.yTicks.map((tick) => (
                                        <g key={`y-${tick.y}`}>
                                            <line
                                                x1={CHART_MARGIN.left}
                                                y1={tick.y}
                                                x2={CHART_MARGIN.left + CHART_INNER_WIDTH}
                                                y2={tick.y}
                                                className="grid-line"
                                            />
                                            <text x={CHART_MARGIN.left - 8} y={tick.y + 3} className="tick-label" textAnchor="end">
                                                {tick.label}
                                            </text>
                                        </g>
                                    ))}
                                    {chartData.xTicks.map((tick) => (
                                        <g key={`x-${tick.x}`}>
                                            <line
                                                x1={tick.x}
                                                y1={CHART_MARGIN.top}
                                                x2={tick.x}
                                                y2={CHART_MARGIN.top + CHART_INNER_HEIGHT}
                                                className="grid-line"
                                            />
                                            <text x={tick.x} y={CHART_MARGIN.top + CHART_INNER_HEIGHT + 16} className="tick-label" textAnchor={tick.anchor}>
                                                {tick.label}
                                            </text>
                                        </g>
                                    ))}
                                    {chartData.trainingPath ? <polyline className="chart-line chart-line-train" points={chartData.trainingPath} /> : null}
                                </>
                            ) : null}
                            <text x="16" y="120" className="axis-label y-label" transform="rotate(-90, 16, 120)">Rolling Average Return</text>
                            <text x={CHART_MARGIN.left + CHART_INNER_WIDTH / 2} y="258" className="axis-label">Episode</text>
                        </svg>
                        <p className="caption">Training rolling average (window=50) across all episodes.</p>
                    </section>
                </main>
            )}
        </div>
    )
}

export default App
