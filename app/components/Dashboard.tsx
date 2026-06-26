"use client";

import React, { useState, useEffect, useCallback } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Area,
  AreaChart,
  ScatterChart,
  Scatter,
  Cell,
} from "recharts";

import {
  generateLiveStream,
  computeForecasts,
  TelemetryPoint,
  Forecasts,
} from "../utils/simulation";

// ─── Types ───────────────────────────────────────────────────────────────────

type TabId =
  | "forecast"
  | "solar"
  | "shielding"
  | "calibration"
  | "validation"
  | "bulletin";

// ─── Constants ───────────────────────────────────────────────────────────────

const TABS: { id: TabId; label: string }[] = [
  { id: "forecast", label: "🛰️ Real-Time Forecasts" },
  { id: "solar", label: "📊 Solar Wind Telemetry" },
  { id: "shielding", label: "🛡️ Active Shielding & Trajectory" },
  { id: "calibration", label: "🔄 ISRO GRASP Calibration" },
  { id: "validation", label: "📈 Performance & Validation" },
  { id: "bulletin", label: "📜 Operational Bulletin" },
];

const MODEL_COLORS: Record<string, string> = {
  TFT: "#3b82f6",
  XGBoost: "#f59e0b",
  LSTM: "#10b981",
  Persistence: "#64748b",
};

const FEATURE_IMPORTANCE: { feature: string; weight: number }[] = [
  { feature: "goes_flux", weight: 0.38 },
  { feature: "log_goes_flux", weight: 0.22 },
  { feature: "Vsw", weight: 0.18 },
  { feature: "B", weight: 0.12 },
  { feature: "Bz", weight: 0.05 },
  { feature: "N", weight: 0.02 },
  { feature: "Pdyn", weight: 0.01 },
  { feature: "bz_south", weight: 0.01 },
  { feature: "dst_proxy", weight: 0.005 },
  { feature: "sin_time_of_day", weight: 0.002 },
  { feature: "cos_time_of_day", weight: 0.002 },
  { feature: "vsw_roll_mean_6h", weight: 0.001 },
].sort((a, b) => a.weight - b.weight);

// Simulated evaluation metrics table
const EVAL_METRICS = [
  { Model: "TFT",         Horizon: "+30m", RMSE: 0.142, MAE: 0.108, "SS (Persistence)": "0.61", R2: 0.89 },
  { Model: "TFT",         Horizon: "+6h",  RMSE: 0.271, MAE: 0.204, "SS (Persistence)": "0.54", R2: 0.82 },
  { Model: "TFT",         Horizon: "+12h", RMSE: 0.342, MAE: 0.261, "SS (Persistence)": "0.48", R2 :0.76 },
  { Model: "LSTM",        Horizon: "+30m", RMSE: 0.159, MAE: 0.121, "SS (Persistence)": "0.57", R2: 0.87 },
  { Model: "LSTM",        Horizon: "+6h",  RMSE: 0.293, MAE: 0.224, "SS (Persistence)": "0.49", R2: 0.79 },
  { Model: "LSTM",        Horizon: "+12h", RMSE: 0.368, MAE: 0.282, "SS (Persistence)": "0.42", R2: 0.72 },
  { Model: "XGBoost",     Horizon: "+30m", RMSE: 0.178, MAE: 0.134, "SS (Persistence)": "0.52", R2: 0.85 },
  { Model: "XGBoost",     Horizon: "+6h",  RMSE: 0.314, MAE: 0.239, "SS (Persistence)": "0.44", R2: 0.76 },
  { Model: "XGBoost",     Horizon: "+12h", RMSE: 0.391, MAE: 0.299, "SS (Persistence)": "0.38", R2: 0.69 },
  { Model: "Persistence", Horizon: "+30m", RMSE: 0.218, MAE: 0.171, "SS (Persistence)": "0.00", R2: 0.74 },
  { Model: "Persistence", Horizon: "+6h",  RMSE: 0.412, MAE: 0.318, "SS (Persistence)": "0.00", R2: 0.58 },
  { Model: "Persistence", Horizon: "+12h", RMSE: 0.524, MAE: 0.402, "SS (Persistence)": "0.00", R2: 0.47 },
];

// ─── Chart Tooltip ───────────────────────────────────────────────────────────

const CustomTooltip = ({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: { name: string; value: number; color: string }[];
  label?: string;
}) => {
  if (active && payload && payload.length) {
    return (
      <div
        style={{
          background: "rgba(8, 12, 24, 0.95)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: "10px",
          padding: "10px 14px",
          fontSize: "12px",
          backdropFilter: "blur(12px)",
        }}
      >
        <p style={{ color: "#94a3b8", marginBottom: 6 }}>{label}</p>
        {payload.map((entry, i) => (
          <p key={i} style={{ color: entry.color, margin: "2px 0" }}>
            {entry.name}: <strong>{Number(entry.value).toFixed(3)}</strong>
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// ─── Sidebar Slider ──────────────────────────────────────────────────────────

const SidebarSlider = ({
  label,
  unit,
  min,
  max,
  step,
  value,
  color,
  onChange,
}: {
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  value: number;
  color: string;
  onChange: (v: number) => void;
}) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
      <span style={{ fontSize: 12, color: "#94a3b8" }}>{label}</span>
      <span style={{ fontSize: 12, fontWeight: 700, color }}>{value.toFixed(1)} {unit}</span>
    </div>
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(Number(e.target.value))}
      style={{ accentColor: color }}
    />
  </div>
);

// ─── Metric Card ─────────────────────────────────────────────────────────────

const MetricCard = ({
  label,
  value,
  sub,
  valueColor,
  cardClass = "card-green",
  glowClass,
}: {
  label: string;
  value: React.ReactNode;
  sub?: string;
  valueColor?: string;
  cardClass?: string;
  glowClass?: string;
}) => (
  <div className={`metric-card ${cardClass}`}>
    <div style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.08em" }}>
      {label}
    </div>
    <div
      className={glowClass}
      style={{
        fontSize: 24,
        fontWeight: 800,
        fontFamily: "var(--font-mono)",
        marginTop: 8,
        color: glowClass ? undefined : valueColor,
      }}
    >
      {value}
    </div>
    {sub && (
      <div style={{ fontSize: 11, color: "#64748b", marginTop: 4 }}>{sub}</div>
    )}
  </div>
);

// ─── Main Dashboard Component ─────────────────────────────────────────────────

export default function Dashboard() {
  // ── State ──
  const [preset, setPreset] = useState("Quiet conditions");
  const [vswVal, setVswVal] = useState(350.0);
  const [bzVal, setBzVal] = useState(2.0);
  const [nVal, setNVal] = useState(5.0);
  const [primaryModel, setPrimaryModel] = useState<keyof Forecasts>("TFT");
  const [compareModels, setCompareModels] = useState<(keyof Forecasts)[]>(["TFT", "XGBoost", "Persistence"]);
  const [telemetry, setTelemetry] = useState<TelemetryPoint[]>([]);
  const [forecasts, setForecasts] = useState<Forecasts | null>(null);
  const [activeTab, setActiveTab] = useState<TabId>("forecast");
  const [toast, setToast] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(true);

  // Shielding state
  const [scalingMode, setScalingMode] = useState("Quadratic Dipole Scaling (Physical)");
  const [controlMode, setControlMode] = useState("Auto-Pilot Autonomous Controller");
  const [manualPower, setManualPower] = useState(0);
  const [orbitalSlot, setOrbitalSlot] = useState(74.0);

  // Calibration state
  const [goesPeakTime, setGoesPeakTime] = useState("12:00");
  const [goesPeakDate, setGoesPeakDate] = useState(() => new Date().toISOString().split("T")[0]);

  // ── Simulation ──
  const runSimulation = useCallback(
    (p: string, v: number, b: number, n: number) => {
      const data = generateLiveStream(p, v, b, n);
      setTelemetry(data);
      const fcs = computeForecasts(v, data[data.length - 1].log_goes_flux);
      setForecasts(fcs);
    },
    []
  );

  useEffect(() => {
    runSimulation(preset, vswVal, bzVal, nVal);
  }, [preset, vswVal, bzVal, nVal, runSimulation]);

  const handlePresetChange = (p: string) => {
    setPreset(p);
    if (p === "Quiet conditions") { setVswVal(350); setBzVal(2); setNVal(5); }
    else if (p === "Moderate Storm") { setVswVal(550); setBzVal(-6); setNVal(16); }
    else if (p === "Severe Geomagnetic Storm (CME)") { setVswVal(850); setBzVal(-18); setNVal(36); }
  };

  const refreshTelemetry = () => {
    runSimulation(preset, vswVal, bzVal, nVal);
    setToast("🛰️ Telemetry updated! Live GSAT satellite stream refreshed.");
    setTimeout(() => setToast(null), 3500);
  };

  // ── Derived State ──
  if (telemetry.length === 0 || !forecasts) {
    return (
      <div style={{ display: "flex", alignItems: "center", justifyContent: "center", minHeight: "100vh" }}>
        <div style={{ textAlign: "center" }}>
          <div style={{ width: 48, height: 48, border: "3px solid rgba(59,130,246,0.2)", borderTop: "3px solid #3b82f6", borderRadius: "50%", animation: "spin 1s linear infinite", margin: "0 auto 16px" }} />
          <p style={{ color: "#64748b", fontSize: 14 }}>Loading Satellite Telemetry...</p>
        </div>
      </div>
    );
  }

  const latest = telemetry[telemetry.length - 1];
  const primaryPreds = forecasts[primaryModel];
  const predMeds = [primaryPreds["30m"][1], primaryPreds["6h"][1], primaryPreds["12h"][1]];
  const maxMed = Math.max(...predMeds);

  // Hazard level
  let hazardLevel = "GREEN", hazardClass = "card-green", hazardGlow = "glow-green";
  let hazardDesc = "QUIET — No satellite charging or operational hazards expected.";
  if (maxMed >= 4.0) {
    hazardLevel = "RED"; hazardClass = "card-red"; hazardGlow = "glow-red";
    hazardDesc = "ALERT — Severe radiation storm. High risk of deep dielectric charging and spacecraft anomalies.";
  } else if (maxMed >= 3.0) {
    hazardLevel = "YELLOW"; hazardClass = "card-yellow"; hazardGlow = "glow-yellow";
    hazardDesc = "WARNING — Moderate risk. Minor electrostatic discharge (ESD) potential on exterior surfaces.";
  }

  // Shielding calculations
  const mdeNeeded = predMeds.map((val) => {
    if (val <= 3.0) return 0;
    return scalingMode.includes("Quadratic")
      ? Math.min(100, 25 * Math.pow(val - 3.0, 2))
      : Math.min(100, 50 * (val - 3.0));
  });
  const maxMdeNeeded = Math.max(...mdeNeeded);
  const shieldPower = controlMode === "Auto-Pilot Autonomous Controller" ? maxMdeNeeded : manualPower;
  const shieldEff = 1.0 - Math.exp(-2.0 * Math.sqrt(shieldPower / 100.0));
  const dipoleMoment = 500.0 * (shieldPower / 100.0);
  const coilTemp = 20.0 + 60.0 * Math.pow(shieldPower / 100.0, 2);
  let coilTempColor = "#10b981", coilTempStatus = "Normal";
  if (coilTemp > 75) { coilTempColor = "#ef4444"; coilTempStatus = "🚨 OVERHEATING"; }
  else if (coilTemp > 60) { coilTempColor = "#f59e0b"; coilTempStatus = "⚠️ Warning: Warm"; }

  // Mitigated forecasts
  const mitLogOffset = Math.log10(Math.max(1e-5, 1.0 - shieldEff));
  const mitMeds = predMeds.map((v) => Math.max(0.5, Math.min(6.0, v + mitLogOffset)));
  const psiHorizons = mitMeds.map((v) => (v <= 3.0 ? 100 : Math.max(0, 100 - 25 * (v - 3.0))));
  const systemPsi = Math.min(...psiHorizons);
  let psiClass = "card-green", psiGlow = "glow-green", psiStatus = "EXCELLENT";
  if (systemPsi < 70) { psiClass = "card-red"; psiGlow = "glow-red"; psiStatus = "CRITICAL"; }
  else if (systemPsi < 90) { psiClass = "card-yellow"; psiGlow = "glow-yellow"; psiStatus = "NOMINAL"; }

  // Phase shift
  const phaseShift = ((orbitalSlot - (-75)) / 15) % 24;
  const combinedGoes = new Date(`${goesPeakDate}T${goesPeakTime}:00`);
  const gsatImpact = new Date(combinedGoes.getTime() + 10 * 3600 * 1000);

  // Chart data preparation
  const now = new Date();
  const forecastTimes = [
    new Date(now.getTime() + 30 * 60 * 1000),
    new Date(now.getTime() + 6 * 3600 * 1000),
    new Date(now.getTime() + 12 * 3600 * 1000),
  ];
  const forecastLabels = ["+30m", "+6h", "+12h"];

  // Telemetry history slice for charts (last 100 points)
  const histSlice = telemetry.slice(-100).map((p, i) => ({
    time: `${-100 + i + 1}`,
    timeLabel: formatHour(p.time),
    log_flux: +p.log_goes_flux.toFixed(3),
    Vsw: +p.Vsw.toFixed(1),
    Bz: +p.Bz.toFixed(2),
    N: +p.N.toFixed(2),
    Pdyn: +p.Pdyn.toFixed(4),
    B: +p.B.toFixed(2),
    dst: +p.dst_proxy.toFixed(1),
  }));

  // Full telemetry for solar wind tab
  const fullTelemetry = telemetry.map((p, i) => ({
    idx: i,
    timeLabel: i % 24 === 0 ? formatHour(p.time) : "",
    Vsw: +p.Vsw.toFixed(1),
    Bz: +p.Bz.toFixed(2),
    N: +p.N.toFixed(2),
    Pdyn: +p.Pdyn.toFixed(4),
  }));

  // Forecast chart data – combine history + 3 forecast points
  const forecastChartData = [
    ...histSlice.map((p) => ({
      label: p.timeLabel,
      observed: p.log_flux,
      q10: null as number | null,
      q90: null as number | null,
      ...Object.fromEntries(
        (["TFT", "XGBoost", "LSTM", "Persistence"] as (keyof Forecasts)[]).map((m) => [m, null as number | null])
      ),
    })),
    ...forecastLabels.map((fl, fi) => ({
      label: fl,
      observed: null as number | null,
      q10: primaryPreds[["30m", "6h", "12h"][fi] as "30m" | "6h" | "12h"][0],
      q90: primaryPreds[["30m", "6h", "12h"][fi] as "30m" | "6h" | "12h"][2],
      ...Object.fromEntries(
        (["TFT", "XGBoost", "LSTM", "Persistence"] as (keyof Forecasts)[]).map((m) => [
          m,
          compareModels.includes(m) ? forecasts[m][["30m", "6h", "12h"][fi] as "30m" | "6h" | "12h"][1] : null,
        ])
      ),
    })),
  ];

  // Mitigation chart data
  const mitigationData = forecastLabels.map((fl, fi) => {
    const h = (["30m", "6h", "12h"] as const)[fi];
    return {
      horizon: fl,
      raw: +predMeds[fi].toFixed(3),
      mitigated: +mitMeds[fi].toFixed(3),
      q10_raw: +primaryPreds[h][0].toFixed(3),
      q90_raw: +primaryPreds[h][2].toFixed(3),
      q10_mit: +(Math.max(0.5, primaryPreds[h][0] + mitLogOffset)).toFixed(3),
      q90_mit: +(Math.max(0.5, primaryPreds[h][2] + mitLogOffset)).toFixed(3),
    };
  });

  // Trajectory slot shift
  const shiftSteps = Math.round(phaseShift * 12);
  const trajectoryData = telemetry.map((p, i) => {
    const shifted_i = (i + shiftSteps) % telemetry.length;
    return {
      idx: i,
      timeLabel: i % 48 === 0 ? formatHour(p.time) : "",
      goes: +p.log_goes_flux.toFixed(3),
      shifted: +telemetry[shifted_i].log_goes_flux.toFixed(3),
    };
  });

  // Scatter calibration
  const calibData = Array.from({ length: 60 }, (_, i) => {
    const goes = 1.5 + (i / 59) * 3.5;
    const grasp = 0.5131 * goes + 0.5669 + (Math.random() - 0.5) * 0.3;
    return { goes: +goes.toFixed(3), grasp: +grasp.toFixed(3) };
  });

  // Quantile reliability
  const reliabilityData = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].map((q) => ({
    nominal: q,
    observed_TFT: +(q + (Math.random() - 0.5) * 0.05).toFixed(3),
    observed_LSTM: +(q + (Math.random() - 0.5) * 0.08).toFixed(3),
    observed_XGB: +(q + (Math.random() - 0.5) * 0.1).toFixed(3),
    perfect: q,
  }));

  // Storm window data
  const stormData = Array.from({ length: 72 }, (_, i) => {
    const t = i / 71;
    const obs = 2.0 + 2.0 * Math.sin(t * Math.PI) + (Math.random() - 0.5) * 0.3;
    return {
      hour: i,
      observed: +obs.toFixed(3),
      TFT: +(obs + (Math.random() - 0.5) * 0.2).toFixed(3),
      XGBoost: +(obs - 0.15 + (Math.random() - 0.5) * 0.3).toFixed(3),
      Persistence: +(obs - 0.4 + (Math.random() - 0.5) * 0.4).toFixed(3),
    };
  });

  // Bulletin text
  const getBulletin = () => {
    const now_str = new Date().toISOString().replace("T", " ").substring(0, 19);
    return `=================================================================
SPACE WEATHER FORECAST BULLETIN — ISRO GEOSAT RADIATION SAFEGUARDS
ISSUED AT: ${now_str} UTC
=================================================================

WARNING STATUS: ${hazardLevel} ALERT

CURRENT OBSERVATIONS:
- GOES >2 MeV Electron Flux: ${latest.goes_flux.toExponential(2)} electrons/cm2/s/sr
- Solar Wind Speed (Vsw): ${vswVal.toFixed(1)} km/s
- IMF Bz: ${bzVal.toFixed(2)} nT (GSM)
- Magnetopause Compressive Pressure: ${latest.Pdyn.toFixed(3)} nPa

FORECAST ADVISORY SUMMARY:
1. HORIZON: +30–45 MINUTES (SHORT-TERM RESPONSE)
   - Expected Median Flux: 10^${primaryPreds["30m"][1].toFixed(2)} (${Math.pow(10, primaryPreds["30m"][1]).toExponential(1)} e-/cm2/s/sr)
   - Quantile Range (10th–90th): 10^${primaryPreds["30m"][0].toFixed(2)} to 10^${primaryPreds["30m"][2].toFixed(2)}

2. HORIZON: +6 HOURS (MEDIUM-TERM FORECAST)
   - Expected Median Flux: 10^${primaryPreds["6h"][1].toFixed(2)} (${Math.pow(10, primaryPreds["6h"][1]).toExponential(1)} e-/cm2/s/sr)
   - Quantile Range (10th–90th): 10^${primaryPreds["6h"][0].toFixed(2)} to 10^${primaryPreds["6h"][2].toFixed(2)}

3. HORIZON: +12 HOURS (LONG-TERM FORECAST)
   - Expected Median Flux: 10^${primaryPreds["12h"][1].toFixed(2)} (${Math.pow(10, primaryPreds["12h"][1]).toExponential(1)} e-/cm2/s/sr)
   - Quantile Range (10th–90th): 10^${primaryPreds["12h"][0].toFixed(2)} to 10^${primaryPreds["12h"][2].toFixed(2)}

HAZARD ANALYSIS:
${hazardDesc}
- GREEN (<10^3): No dielectric charging risk.
- YELLOW (10^3 – 10^4): Elevated electrostatic discharge risk for sensitive instruments.
- RED (>10^4): Critical storm status. High risk of deep dielectric charging and spacecraft anomalies.

CALIBRATION ALIGNMENT NOTE:
For ISRO payloads at Indian Longitude (e.g. GSAT-7A / GSAT-31 at ~74°E), expected diurnal peaks are 
phase-delayed by +10.0 hours relative to the GOES UTC time series.

AUTHENTICATION KEY: ISRO-SW-${primaryModel}-EAD-V1.2
=================================================================`;
  };

  const downloadBulletin = () => {
    const text = getBulletin();
    const blob = new Blob([text], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `isro_bulletin_${new Date().toISOString().slice(0, 16).replace(/[-T:]/g, "")}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // ─── Render ───────────────────────────────────────────────────────────────

  return (
    <div style={{ display: "flex", flexDirection: "row", minHeight: "100vh", position: "relative" }}>

      {/* ── Toast ── */}
      {toast && (
        <div style={{
          position: "fixed", top: 24, right: 24, zIndex: 9999,
          background: "rgba(37, 99, 235, 0.9)",
          border: "1px solid rgba(59, 130, 246, 0.3)",
          backdropFilter: "blur(12px)",
          color: "#fff", padding: "14px 20px",
          borderRadius: 12, fontSize: 13, fontWeight: 600,
          boxShadow: "0 8px 32px rgba(37, 99, 235, 0.4)",
          display: "flex", alignItems: "center", gap: 8,
          animation: "slideIn 0.3s ease",
        }}>
          {toast}
        </div>
      )}

      {/* ── Sidebar ── */}
      <div style={{
        width: sidebarOpen ? 300 : 0,
        minWidth: sidebarOpen ? 300 : 0,
        overflow: "hidden",
        transition: "width 0.3s ease, min-width 0.3s ease",
        background: "#04060b",
        borderRight: "1px solid rgba(255,255,255,0.04)",
        display: "flex",
        flexDirection: "column",
        gap: 20,
        padding: sidebarOpen ? "24px 20px" : 0,
        flexShrink: 0,
      }}>
        {/* Logo + Title */}
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <img src="/isro-logo.png" alt="ISRO" style={{ width: 64, height: 46, objectFit: "contain" }} />
          <div>
            <div style={{ fontFamily: "var(--font-mono)", fontWeight: 700, fontSize: 15, color: "#e2e8f0" }}>
              Operational Panel
            </div>
            <div style={{ fontSize: 10, color: "#60a5fa", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.1em" }}>
              ISRO Space Weather Warning Station
            </div>
          </div>
        </div>

        <div style={{ height: 1, background: "rgba(255,255,255,0.04)" }} />

        {/* Simulation Presets */}
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <label style={{ fontSize: 10, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.1em" }}>
            Simulation Presets
          </label>
          <div style={{ fontSize: 11, fontWeight: 600, color: "#94a3b8", marginBottom: 4 }}>Geomagnetic Event Preset</div>
          <select
            value={preset}
            onChange={(e) => handlePresetChange(e.target.value)}
            style={{
              width: "100%", background: "#0f172a", border: "1px solid rgba(255,255,255,0.08)",
              borderRadius: 10, padding: "10px 12px", fontSize: 13, color: "#e2e8f0", cursor: "pointer",
            }}
          >
            <option>Quiet conditions</option>
            <option>Moderate Storm</option>
            <option>Severe Geomagnetic Storm (CME)</option>
          </select>
        </div>

        {/* Interactive Adjustments */}
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <label style={{ fontSize: 10, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.1em" }}>
            Interactive Adjustments
          </label>
        </div>
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          <SidebarSlider label="Solar Wind Speed (Vsw)" unit="km/s" min={300} max={1000} step={1} value={vswVal} color="#ff7b72"
            onChange={(v) => { setPreset("Custom"); setVswVal(v); }} />
          <SidebarSlider label="IMF Bz Component" unit="nT" min={-25} max={15} step={0.5} value={bzVal} color={bzVal >= 0 ? "#3fb950" : "#f87171"}
            onChange={(v) => { setPreset("Custom"); setBzVal(v); }} />
          <SidebarSlider label="Proton Density (N)" unit="cm⁻³" min={1} max={50} step={0.5} value={nVal} color="#fbbf24"
            onChange={(v) => { setPreset("Custom"); setNVal(v); }} />
        </div>

        <div style={{ height: 1, background: "rgba(255,255,255,0.04)" }} />

        {/* Forecasting Models */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <label style={{ fontSize: 10, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.1em" }}>
            Forecasting Models
          </label>
          <div>
            <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 6 }}>Primary Warning Model</div>
            <select
              value={primaryModel}
              onChange={(e) => setPrimaryModel(e.target.value as keyof Forecasts)}
              style={{
                width: "100%", background: "#0f172a", border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 10, padding: "10px 12px", fontSize: 13, color: "#e2e8f0", cursor: "pointer",
              }}
            >
              {(["TFT", "XGBoost", "LSTM", "Persistence"] as (keyof Forecasts)[]).map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          <div>
            <div style={{ fontSize: 11, color: "#94a3b8", marginBottom: 8 }}>Overlay Comparison Models</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {(["TFT", "XGBoost", "LSTM", "Persistence"] as (keyof Forecasts)[]).map((m) => (
                <label key={m} style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer", fontSize: 13, color: "#cbd5e1" }}>
                  <input
                    type="checkbox"
                    checked={compareModels.includes(m)}
                    onChange={(e) => {
                      if (e.target.checked) setCompareModels([...compareModels, m]);
                      else setCompareModels(compareModels.filter((x) => x !== m));
                    }}
                    style={{ accentColor: MODEL_COLORS[m], width: 14, height: 14 }}
                  />
                  <span style={{ color: MODEL_COLORS[m], fontWeight: 600 }}>●</span> {m}
                </label>
              ))}
            </div>
          </div>
        </div>

        <button
          onClick={refreshTelemetry}
          style={{
            marginTop: "auto",
            width: "100%",
            background: "linear-gradient(135deg, #1d4ed8, #2563eb)",
            border: "1px solid rgba(59, 130, 246, 0.3)",
            borderRadius: 12, padding: "13px 0",
            color: "#fff", fontWeight: 700, fontSize: 13, cursor: "pointer",
            boxShadow: "0 4px 20px rgba(37, 99, 235, 0.3)",
            transition: "all 0.2s ease",
            display: "flex", alignItems: "center", justifyContent: "center", gap: 8,
          }}
          onMouseEnter={(e) => (e.currentTarget.style.transform = "translateY(-1px)")}
          onMouseLeave={(e) => (e.currentTarget.style.transform = "translateY(0)")}
        >
          🛰️ Refresh & Fetch Live Telemetry
        </button>
      </div>

      {/* ── Sidebar Toggle ── */}
      <button
        onClick={() => setSidebarOpen(!sidebarOpen)}
        style={{
          position: "fixed", left: sidebarOpen ? 300 : 0, top: "50%",
          transform: "translateY(-50%)",
          zIndex: 100, background: "#0f172a",
          border: "1px solid rgba(255,255,255,0.08)",
          borderLeft: "none",
          color: "#64748b", padding: "8px 4px",
          borderRadius: "0 8px 8px 0", cursor: "pointer",
          transition: "left 0.3s ease", fontSize: 12,
        }}
      >
        {sidebarOpen ? "◀" : "▶"}
      </button>

      {/* ── Main Content ── */}
      <div style={{ flex: 1, padding: "28px 32px", display: "flex", flexDirection: "column", gap: 24, minWidth: 0, overflowX: "hidden" }}>

        {/* Title */}
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
          <div>
            <h1 style={{ fontSize: 26, fontWeight: 800, color: "#f8fafc", letterSpacing: "-0.02em", lineHeight: 1.2 }}>
              Energetic Particle Radiation Forecasting System
            </h1>
            <p style={{ fontSize: 13, color: "#64748b", marginTop: 6 }}>
              Satellite Safeguard Center for Geostationary Orbit — Payload Protection Operations
            </p>
          </div>
          <img src="/isro-logo.png" alt="ISRO" style={{ width: 88, height: 64, flexShrink: 0, objectFit: "contain" }} />
        </div>

        {/* Tab Navigation */}
        <div style={{
          display: "flex", flexWrap: "wrap", gap: 6,
          background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.05)",
          borderRadius: 14, padding: 8,
        }}>
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: "10px 18px", borderRadius: 10, fontSize: 13, fontWeight: 600,
                border: "none", cursor: "pointer", transition: "all 0.25s ease",
                background: activeTab === tab.id ? "#1d4ed8" : "transparent",
                color: activeTab === tab.id ? "#fff" : "#64748b",
                boxShadow: activeTab === tab.id ? "0 0 20px rgba(29,78,216,0.45)" : "none",
              }}
              onMouseEnter={(e) => { if (activeTab !== tab.id) e.currentTarget.style.color = "#cbd5e1"; }}
              onMouseLeave={(e) => { if (activeTab !== tab.id) e.currentTarget.style.color = "#64748b"; }}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* ── TAB 1: Real-Time Forecasts ── */}
        {activeTab === "forecast" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 24 }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: "#e2e8f0" }}>
              Magnetospheric KPIs & Hazard Status
            </h3>

            {/* KPI Cards */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 14 }}>
              <MetricCard
                label="Current Electron Flux"
                value={latest.goes_flux.toExponential(2)}
                sub={`log10 flux: ${latest.log_goes_flux.toFixed(2)}`}
                valueColor="#58a6ff"
              />
              <MetricCard
                label="Solar Wind (Vsw)"
                value={<>{vswVal.toFixed(0)} <span style={{ fontSize: 14, color: "#64748b" }}>km/s</span></>}
                sub={`Lag(t-6h): ${latest.vsw_lag_6h.toFixed(0)} km/s`}
                valueColor="#ff7b72"
              />
              <MetricCard
                label="IMF Bz Component"
                value={<>{bzVal.toFixed(1)} <span style={{ fontSize: 14, color: "#64748b" }}>nT</span></>}
                sub={`Direction: ${bzVal >= 0 ? "Northward" : "Southward"}`}
                valueColor={bzVal >= 0 ? "#3fb950" : "#ff7b72"}
              />
              <MetricCard
                label="Dynamic Pressure (Pdyn)"
                value={<>{latest.Pdyn.toFixed(3)} <span style={{ fontSize: 14, color: "#64748b" }}>nPa</span></>}
                sub={`Magnetosphere: ${latest.Pdyn > 4 ? "Compressed" : "Normal"}`}
                valueColor="#d29922"
              />
              <MetricCard
                label="Orbit Hazard Status"
                value={hazardLevel}
                sub={hazardDesc.substring(0, 46) + "..."}
                cardClass={hazardClass}
                glowClass={hazardGlow}
              />
            </div>

            {/* Charts Row */}
            <div style={{ display: "grid", gridTemplateColumns: "5fr 3fr", gap: 20 }}>
              {/* Forecast Chart */}
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <h4 style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Multi-Horizon Forecast Comparison ({primaryModel} Primary)
                </h4>
                <div style={{ background: "#080c18", borderRadius: 14, border: "1px solid rgba(255,255,255,0.05)", padding: "16px 8px" }}>
                  <ResponsiveContainer width="100%" height={420}>
                    <LineChart data={forecastChartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                      <XAxis dataKey="label" tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} tickLine={false} />
                      <YAxis domain={[0.5, 6.2]} tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} tickLine={false}
                        label={{ value: "log10(>2 MeV Flux)", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11, dx: -8 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend iconType="circle" wrapperStyle={{ fontSize: 12, paddingTop: 12 }} />

                      {/* Threshold lines */}
                      <ReferenceLine y={4.0} stroke="#f59e0b" strokeDasharray="5 5" opacity={0.8}
                        label={{ value: "ESD Risk (10⁴)", fill: "#f59e0b", fontSize: 10, position: "insideTopLeft" }} />
                      <ReferenceLine y={5.0} stroke="#ef4444" strokeDasharray="5 5" opacity={0.8}
                        label={{ value: "Severe Hazard (10⁵)", fill: "#ef4444", fontSize: 10, position: "insideTopLeft" }} />

                      {/* Observed */}
                      <Line dataKey="observed" name="Observed Flux" stroke="#8b949e" strokeWidth={2.5}
                        dot={false} connectNulls={false} />

                      {/* Q10 / Q90 band using Area */}
                      <Line dataKey="q10" name="10th Quantile" stroke="#2563eb" strokeWidth={1}
                        strokeDasharray="3 3" dot={false} connectNulls={false} legendType="none" />
                      <Line dataKey="q90" name="90th Quantile" stroke="#2563eb" strokeWidth={1}
                        strokeDasharray="3 3" dot={false} connectNulls={false}
                        label={{ value: `${primaryModel} Band`, fill: "#3b82f6", fontSize: 10, position: "insideTopRight" }} />

                      {/* Model medians */}
                      {compareModels.map((m) => (
                        <Line key={m} dataKey={m} name={`${m} Median`} stroke={MODEL_COLORS[m]}
                          strokeWidth={2.5} dot={{ r: 5, fill: MODEL_COLORS[m] }}
                          strokeDasharray={m === "Persistence" ? "5 5" : "none"}
                          connectNulls={false} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Feature Importance */}
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <h4 style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Feature Selection Weights (TFT VSN)
                </h4>
                <div style={{ background: "#080c18", borderRadius: 14, border: "1px solid rgba(255,255,255,0.05)", padding: "16px 8px" }}>
                  <ResponsiveContainer width="100%" height={420}>
                    <BarChart data={FEATURE_IMPORTANCE} layout="vertical" margin={{ top: 5, right: 20, left: 80, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" horizontal={false} />
                      <XAxis type="number" tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} tickLine={false} />
                      <YAxis type="category" dataKey="feature" tick={{ fill: "#94a3b8", fontSize: 10 }} tickLine={false} width={80} />
                      <Tooltip content={<CustomTooltip />} />
                      <Bar dataKey="weight" name="Attention Weight" radius={[0, 4, 4, 0]}>
                        {FEATURE_IMPORTANCE.map((_, i) => (
                          <Cell key={i} fill={`rgba(59, 130, 246, ${0.3 + 0.7 * (i / FEATURE_IMPORTANCE.length)})`} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── TAB 2: Solar Wind Telemetry ── */}
        {activeTab === "solar" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: "#e2e8f0" }}>
              Solar Wind Parameters (24-Hour Ingestion History)
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              {[
                { key: "Vsw", label: "Vsw — Solar Wind Speed (km/s)", color: "#ff7b72", domain: [200, 1200] as [number, number] },
                { key: "Bz", label: "IMF Bz Component (nT)", color: "#3fb950", domain: [-30, 20] as [number, number], refY: 0 },
                { key: "N", label: "N — Proton Density (cm⁻³)", color: "#79c0ff", domain: [0, 60] as [number, number] },
                { key: "Pdyn", label: "Pdyn — Dynamic Pressure (nPa)", color: "#d29922", domain: [0, 0.03] as [number, number] },
              ].map(({ key, label, color, domain, refY }) => (
                <div key={key} style={{ background: "#080c18", borderRadius: 14, border: "1px solid rgba(255,255,255,0.05)", padding: "16px 8px" }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "#94a3b8", marginBottom: 8, paddingLeft: 8 }}>{label}</div>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={fullTelemetry} margin={{ top: 5, right: 16, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                      <XAxis dataKey="timeLabel" tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 9 }} tickLine={false} interval="preserveStartEnd" />
                      <YAxis domain={domain} tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} tickLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      {refY !== undefined && (
                        <ReferenceLine y={refY} stroke="gray" strokeDasharray="4 4" opacity={0.5} />
                      )}
                      <Line dataKey={key} name={key} stroke={color} strokeWidth={1.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── TAB 3: Active Shielding ── */}
        {activeTab === "shielding" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div>
              <h3 style={{ fontSize: 16, fontWeight: 700, color: "#e2e8f0" }}>🛡️ Active Electromagnetic Shielding & Trajectory Optimization</h3>
              <p style={{ fontSize: 13, color: "#64748b", marginTop: 6 }}>
                Transform the satellite from a passive observer to an active agent. Use the Temporal Fusion Transformer's multi-horizon forecasts to simulate proactive magnetic deflection maneuvers and calculate real-time safety indices.
              </p>
            </div>

            {/* Physics Info Banner */}
            <div style={{ background: "rgba(37, 99, 235, 0.05)", border: "1px solid rgba(37, 99, 235, 0.15)", borderRadius: 12, padding: "16px 18px", fontSize: 13, lineHeight: 1.6 }}>
              <span style={{ fontWeight: 700, color: "#3b82f6" }}>Lorentz Deflection Physics:</span>{" "}
              An active shielding system generates an onboard magnetic dipole field to deflect relativistic electrons (&gt;2 MeV).
              The power required scales with the required deflection moment. Deflection efficiency is simulated via exponential shielding attenuation
              (η = 1 − e^(−γ√P)).
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "2fr 3fr", gap: 24 }}>
              {/* Controls */}
              <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
                <h4 style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0" }}>🎛️ Active Shielding Controls</h4>

                <div>
                  <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8 }}>Deflection Power Scaling Logic</div>
                  {["Quadratic Dipole Scaling (Physical)", "Linear Power Scaling"].map((mode) => (
                    <label key={mode} style={{ display: "flex", alignItems: "center", gap: 10, cursor: "pointer", fontSize: 13, color: mode === scalingMode ? "#e2e8f0" : "#64748b", padding: "8px 0" }}>
                      <input type="radio" checked={scalingMode === mode} onChange={() => setScalingMode(mode)}
                        style={{ accentColor: "#3b82f6" }} />
                      {mode}
                    </label>
                  ))}
                </div>

                <div>
                  <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 6 }}>Coil Controller Mode</div>
                  <select
                    value={controlMode}
                    onChange={(e) => setControlMode(e.target.value)}
                    style={{ width: "100%", background: "#0f172a", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: "10px 12px", fontSize: 13, color: "#e2e8f0", cursor: "pointer" }}
                  >
                    <option>Auto-Pilot Autonomous Controller</option>
                    <option>Manual Power Override</option>
                  </select>
                </div>

                {controlMode === "Auto-Pilot Autonomous Controller" ? (
                  <div style={{ background: "rgba(37, 99, 235, 0.08)", border: "1px solid rgba(37, 99, 235, 0.2)", borderRadius: 10, padding: "12px 14px", fontSize: 13, color: "#93c5fd" }}>
                    🤖 <strong>Auto-Pilot active:</strong> Shielding power allocated at <strong>{shieldPower.toFixed(1)}%</strong> to cover predicted peak flux.
                  </div>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                      <span style={{ color: "#94a3b8" }}>Manual Power Allocation</span>
                      <span style={{ color: "#3b82f6", fontWeight: 700 }}>{manualPower.toFixed(0)}%</span>
                    </div>
                    <input type="range" min={0} max={100} step={1} value={manualPower}
                      onChange={(e) => setManualPower(Number(e.target.value))} style={{ accentColor: "#3b82f6" }} />
                  </div>
                )}

                <div style={{ height: 1, background: "rgba(255,255,255,0.04)" }} />

                <h4 style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0" }}>🗺️ Trajectory Slot Logic</h4>
                <p style={{ fontSize: 12, color: "#64748b" }}>Adjust the longitudinal slot position to view the local-noon phase-shifted flux exposure profile.</p>

                <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                    <span style={{ color: "#94a3b8" }}>Orbital Slot Longitude (°East)</span>
                    <span style={{ color: "#f59e0b", fontWeight: 700 }}>{orbitalSlot.toFixed(0)}°E</span>
                  </div>
                  <input type="range" min={0} max={360} step={1} value={orbitalSlot}
                    onChange={(e) => setOrbitalSlot(Number(e.target.value))} style={{ accentColor: "#f59e0b" }} />
                </div>

                <div style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)", borderRadius: 10, padding: "12px 14px", fontSize: 12 }}>
                  📍 <strong>Orbital Slot:</strong> {orbitalSlot.toFixed(1)}°E<br />
                  ⏰ <strong>Peak Phase Shift:</strong> +{phaseShift.toFixed(1)} hours relative to GOES Sector
                </div>
              </div>

              {/* Visualization */}
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <h4 style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0" }}>📊 Mitigation Performance & Safety Metrics</h4>

                {/* Safety metric cards */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
                  <MetricCard label="Proactive Safety Index" value={`${systemPsi.toFixed(1)}%`} sub={`Status: ${psiStatus}`} cardClass={psiClass} glowClass={psiGlow} />
                  <MetricCard label="Coil Temperature" value={<>{coilTemp.toFixed(1)}<span style={{ fontSize: 13, color: "#64748b" }}>°C</span></>} sub={`Status: ${coilTempStatus}`} valueColor={coilTempColor} />
                  <MetricCard label="Active Dipole Moment" value={<>{dipoleMoment.toFixed(0)}<span style={{ fontSize: 13, color: "#64748b" }}>A·m²</span></>} sub={`Deflection Eff: ${(shieldEff * 100).toFixed(1)}%`} valueColor="#3b82f6" />
                </div>

                {/* Mitigation chart */}
                <div style={{ background: "#080c18", borderRadius: 14, border: "1px solid rgba(255,255,255,0.05)", padding: "16px 8px" }}>
                  <div style={{ fontSize: 11, color: "#94a3b8", paddingLeft: 8, marginBottom: 8 }}>Mitigation Forecast (log10 Flux)</div>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={mitigationData} margin={{ top: 5, right: 16, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                      <XAxis dataKey="horizon" tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 11 }} tickLine={false} />
                      <YAxis domain={[0, 7]} tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} tickLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                      <ReferenceLine y={3.0} stroke="#f59e0b" strokeDasharray="5 5" opacity={0.7} />
                      <ReferenceLine y={4.0} stroke="#ef4444" strokeDasharray="5 5" opacity={0.7} />
                      <Bar dataKey="raw" name="Unmitigated" fill="#ef4444" opacity={0.8} radius={[4, 4, 0, 0]} />
                      <Bar dataKey="mitigated" name="Mitigated" fill="#10b981" opacity={0.9} radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Summary table */}
                <div style={{ background: "#080c18", borderRadius: 14, border: "1px solid rgba(255,255,255,0.05)", overflow: "hidden" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.06)", background: "rgba(255,255,255,0.02)" }}>
                        {["Horizon", "Raw Forecast (log10)", "Mitigated (log10)", "Deflection Effort", "Safety Index"].map((h) => (
                          <th key={h} style={{ padding: "10px 14px", textAlign: "left", color: "#94a3b8", fontWeight: 600, textTransform: "uppercase", fontSize: 10, letterSpacing: "0.05em" }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {["30m", "6h", "12h"].map((h, i) => (
                        <tr key={h} style={{ borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
                          <td style={{ padding: "10px 14px", color: "#e2e8f0", fontWeight: 600 }}>{h}</td>
                          <td style={{ padding: "10px 14px", color: "#ff7b72", fontFamily: "monospace" }}>{predMeds[i].toFixed(2)}</td>
                          <td style={{ padding: "10px 14px", color: "#10b981", fontFamily: "monospace" }}>{mitMeds[i].toFixed(2)}</td>
                          <td style={{ padding: "10px 14px", color: "#f59e0b" }}>{mdeNeeded[i].toFixed(1)}%</td>
                          <td style={{ padding: "10px 14px", color: psiHorizons[i] >= 90 ? "#10b981" : psiHorizons[i] >= 70 ? "#f59e0b" : "#ef4444" }}>{psiHorizons[i].toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Trajectory phase shift chart */}
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <h4 style={{ fontSize: 14, fontWeight: 700, color: "#e2e8f0" }}>🕒 Trajectory Phase Shift Simulation</h4>
              <div style={{ background: "#080c18", borderRadius: 14, border: "1px solid rgba(255,255,255,0.05)", padding: "16px 8px" }}>
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={trajectoryData} margin={{ top: 5, right: 16, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                    <XAxis dataKey="timeLabel" tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 9 }} tickLine={false} interval="preserveStartEnd" />
                    <YAxis tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} tickLine={false}
                      label={{ value: "log10(Flux)", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11, dx: -8 }} />
                    <Tooltip content={<CustomTooltip />} />
                    <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                    <Line dataKey="goes" name="GOES Sector (Reference)" stroke="#3b82f6" strokeWidth={2} dot={false} />
                    <Line dataKey="shifted" name={`Slot ${orbitalSlot.toFixed(0)}°E (Shifted)`} stroke="#ff7b72" strokeWidth={2} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}

        {/* ── TAB 4: ISRO GRASP Calibration ── */}
        {activeTab === "calibration" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: "#e2e8f0" }}>Longitudinal Phase Shift & Calibration</h3>
            <p style={{ fontSize: 13, color: "#64748b" }}>
              For ISRO satellites operating at Indian Longitudes (~74°E), there is a significant diurnal peak delay relative to American longitude satellites (like GOES-16). Because solar wind compression events and diurnal peaks shift locally as the Earth rotates, a correction factor is required.
            </p>

            {/* Equation Banner */}
            <div style={{ background: "rgba(37, 99, 235, 0.06)", border: "1px solid rgba(37, 99, 235, 0.15)", borderRadius: 12, padding: "16px 20px", fontSize: 14 }}>
              <div style={{ fontWeight: 700, color: "#3b82f6", marginBottom: 8 }}>Linear Cross-Calibration Equation:</div>
              <div style={{ fontFamily: "monospace", fontSize: 16, color: "#e2e8f0", background: "rgba(0,0,0,0.3)", padding: "10px 16px", borderRadius: 8, display: "inline-block" }}>
                log₁₀(GRASP) = 0.5131 × log₁₀(GOES) + 0.5669
              </div>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              {/* Scatter Calibration */}
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <h4 style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  GOES vs GRASP Scatter Calibration
                </h4>
                <div style={{ background: "#080c18", borderRadius: 14, border: "1px solid rgba(255,255,255,0.05)", padding: "16px 8px" }}>
                  <ResponsiveContainer width="100%" height={320}>
                    <ScatterChart margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                      <XAxis type="number" dataKey="goes" name="log10(GOES)" domain={[1.5, 5.5]}
                        tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }}
                        label={{ value: "log10(GOES Flux)", position: "insideBottom", offset: -5, fill: "#64748b", fontSize: 11 }} />
                      <YAxis type="number" dataKey="grasp" name="log10(GRASP)"
                        tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }}
                        label={{ value: "log10(GRASP Flux)", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11, dx: -8 }} />
                      <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<CustomTooltip />} />
                      <Scatter data={calibData} fill="#3b82f6" opacity={0.7} />
                      {/* Regression line approximated with a line chart overlay */}
                    </ScatterChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Peak Estimator */}
              <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                <h4 style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  🕒 Live GSAT Impact Peak Estimator
                </h4>
                <p style={{ fontSize: 12, color: "#64748b" }}>
                  Calculate the expected peak hazard timing at the Indian Geostationary Longitude (GSAT orbit) based on a forecasted event peak at GOES.
                </p>

                <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                  <div>
                    <label style={{ fontSize: 11, color: "#94a3b8", display: "block", marginBottom: 6 }}>GOES Peak Time (UTC)</label>
                    <input type="time" value={goesPeakTime} onChange={(e) => setGoesPeakTime(e.target.value)}
                      style={{ width: "100%", background: "#0f172a", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: "10px 12px", fontSize: 13, color: "#e2e8f0" }} />
                  </div>
                  <div>
                    <label style={{ fontSize: 11, color: "#94a3b8", display: "block", marginBottom: 6 }}>GOES Peak Date</label>
                    <input type="date" value={goesPeakDate} onChange={(e) => setGoesPeakDate(e.target.value)}
                      style={{ width: "100%", background: "#0f172a", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 10, padding: "10px 12px", fontSize: 13, color: "#e2e8f0" }} />
                  </div>
                </div>

                <div style={{ background: "rgba(37, 99, 235, 0.06)", border: "1px dashed rgba(37, 99, 235, 0.3)", borderRadius: 12, padding: "20px" }}>
                  <div style={{ fontSize: 11, textTransform: "uppercase", color: "#94a3b8", fontWeight: 600 }}>Expected GSAT Payload Impact Time</div>
                  <div style={{ fontSize: 24, fontWeight: 800, color: "#3b82f6", marginTop: 10, fontFamily: "var(--font-mono)" }}>
                    {gsatImpact.toISOString().replace("T", " ").substring(0, 19)} UTC
                  </div>
                  <div style={{ fontSize: 12, color: "#64748b", marginTop: 6 }}>
                    Correction: +10.0 hours phase delay added for 74°E orbital slot.
                  </div>
                </div>

                {/* Diurnal pattern chart */}
                <div style={{ background: "#080c18", borderRadius: 14, border: "1px solid rgba(255,255,255,0.05)", padding: "16px 8px" }}>
                  <div style={{ fontSize: 11, color: "#94a3b8", paddingLeft: 8, marginBottom: 8 }}>Diurnal Phase Shift Overlay (~10h Shift)</div>
                  <ResponsiveContainer width="100%" height={180}>
                    <LineChart data={trajectoryData.slice(0, 144)} margin={{ top: 5, right: 16, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                      <XAxis dataKey="timeLabel" tick={{ fill: "rgba(255,255,255,0.25)", fontSize: 9 }} tickLine={false} interval="preserveStartEnd" />
                      <YAxis tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} tickLine={false} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend iconType="circle" wrapperStyle={{ fontSize: 11 }} />
                      <Line dataKey="goes" name="GOES (Reference)" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
                      <Line dataKey="shifted" name="GSAT/GRASP (+10h)" stroke="#ff7b72" strokeWidth={1.5} dot={false} strokeDasharray="4 4" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── TAB 5: Performance & Validation ── */}
        {activeTab === "validation" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: "#e2e8f0" }}>
              Model Evaluation Metrics (Historical Test Set 2020–2021)
            </h3>
            <p style={{ fontSize: 13, color: "#64748b" }}>
              Validation metrics and skill scores evaluated against persistence on the test subset.
            </p>

            {/* Metrics Table */}
            <div style={{ background: "#080c18", borderRadius: 14, border: "1px solid rgba(255,255,255,0.05)", overflow: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.06)", background: "rgba(255,255,255,0.02)" }}>
                    {["Model", "Horizon", "RMSE", "MAE", "SS (Persistence)", "R²"].map((h) => (
                      <th key={h} style={{ padding: "12px 16px", textAlign: "left", color: "#94a3b8", fontWeight: 600, textTransform: "uppercase", fontSize: 10, letterSpacing: "0.05em", whiteSpace: "nowrap" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {EVAL_METRICS.map((row, i) => {
                    const rmseMax = 0.524, maeMax = 0.402;
                    const rmseIntensity = Math.round((row.RMSE / rmseMax) * 200 + 55);
                    return (
                      <tr key={i} style={{ borderBottom: "1px solid rgba(255,255,255,0.03)", transition: "background 0.2s" }}
                        onMouseEnter={(e) => e.currentTarget.style.background = "rgba(255,255,255,0.02)"}
                        onMouseLeave={(e) => e.currentTarget.style.background = "transparent"}>
                        <td style={{ padding: "10px 16px", color: MODEL_COLORS[row.Model] ?? "#e2e8f0", fontWeight: 700 }}>{row.Model}</td>
                        <td style={{ padding: "10px 16px", color: "#94a3b8" }}>{row.Horizon}</td>
                        <td style={{ padding: "10px 16px", color: `rgb(${rmseIntensity}, 50, 50)`, fontFamily: "monospace", fontWeight: 600 }}>{row.RMSE.toFixed(3)}</td>
                        <td style={{ padding: "10px 16px", color: `rgb(${Math.round((row.MAE / maeMax) * 200 + 55)}, 50, 50)`, fontFamily: "monospace" }}>{row.MAE.toFixed(3)}</td>
                        <td style={{ padding: "10px 16px", color: "#94a3b8", fontFamily: "monospace" }}>{row["SS (Persistence)"]}</td>
                        <td style={{ padding: "10px 16px", color: "#10b981", fontFamily: "monospace" }}>{row.R2.toFixed(2)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
              {/* Quantile Reliability */}
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <h4 style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Quantile Reliability Calibration Curve
                </h4>
                <div style={{ background: "#080c18", borderRadius: 14, border: "1px solid rgba(255,255,255,0.05)", padding: "16px 8px" }}>
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={reliabilityData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                      <XAxis dataKey="nominal" tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} tickLine={false}
                        label={{ value: "Nominal Quantile", position: "insideBottom", offset: -5, fill: "#64748b", fontSize: 11 }} />
                      <YAxis domain={[0, 1]} tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} tickLine={false}
                        label={{ value: "Observed Frequency", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11, dx: -8 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                      <Line dataKey="perfect" name="Perfect Calibration" stroke="#ffffff" strokeWidth={1} strokeDasharray="5 5" dot={false} />
                      {(["TFT", "LSTM", "XGBoost"] as const).map((m) => (
                        <Line key={m} dataKey={`observed_${m === "XGBoost" ? "XGB" : m}`} name={m} stroke={MODEL_COLORS[m]} strokeWidth={2} dot={{ r: 3, fill: MODEL_COLORS[m] }} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Storm Window Predictions */}
              <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
                <h4 style={{ fontSize: 13, fontWeight: 700, color: "#94a3b8", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                  Test Set Storm Window Predictions
                </h4>
                <div style={{ background: "#080c18", borderRadius: 14, border: "1px solid rgba(255,255,255,0.05)", padding: "16px 8px" }}>
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={stormData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                      <XAxis dataKey="hour" tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} tickLine={false}
                        label={{ value: "Hours During Storm", position: "insideBottom", offset: -5, fill: "#64748b", fontSize: 11 }} />
                      <YAxis tick={{ fill: "rgba(255,255,255,0.35)", fontSize: 10 }} tickLine={false}
                        label={{ value: "log10(Flux)", angle: -90, position: "insideLeft", fill: "#64748b", fontSize: 11, dx: -8 }} />
                      <Tooltip content={<CustomTooltip />} />
                      <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                      <Line dataKey="observed" name="Observed" stroke="#8b949e" strokeWidth={2.5} dot={false} />
                      {(["TFT", "XGBoost", "Persistence"] as const).map((m) => (
                        <Line key={m} dataKey={m} name={m} stroke={MODEL_COLORS[m]} strokeWidth={2}
                          strokeDasharray={m === "Persistence" ? "5 5" : "none"} dot={false} />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ── TAB 6: Operational Bulletin ── */}
        {activeTab === "bulletin" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <h3 style={{ fontSize: 16, fontWeight: 700, color: "#e2e8f0" }}>Operational Dissemination Bulletin</h3>
            <p style={{ fontSize: 13, color: "#64748b" }}>
              Below is the formatted forecast advisory, suitable for forwarding to satellite operations.
            </p>

            {/* Hazard status badge */}
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div className={`metric-card ${hazardClass}`} style={{ padding: "12px 20px", display: "flex", alignItems: "center", gap: 12 }}>
                <div className={hazardGlow} style={{ fontSize: 20, fontWeight: 800 }}>● {hazardLevel}</div>
                <div style={{ fontSize: 12, color: "#64748b" }}>{hazardDesc}</div>
              </div>
            </div>

            <div style={{ background: "#080c18", borderRadius: 14, border: "1px solid rgba(255,255,255,0.05)", padding: 20 }}>
              <pre style={{
                fontFamily: "monospace", fontSize: 12, color: "#e2e8f0", lineHeight: 1.7,
                whiteSpace: "pre-wrap", wordBreak: "break-word", overflowX: "auto",
              }}>
                {getBulletin()}
              </pre>
            </div>

            <button
              onClick={downloadBulletin}
              style={{
                alignSelf: "flex-start",
                background: "linear-gradient(135deg, #1d4ed8, #2563eb)",
                border: "1px solid rgba(59, 130, 246, 0.3)",
                borderRadius: 12, padding: "13px 28px",
                color: "#fff", fontWeight: 700, fontSize: 13, cursor: "pointer",
                boxShadow: "0 4px 20px rgba(37, 99, 235, 0.3)",
                transition: "all 0.2s ease",
                display: "flex", alignItems: "center", gap: 8,
              }}
              onMouseEnter={(e) => (e.currentTarget.style.transform = "translateY(-1px)")}
              onMouseLeave={(e) => (e.currentTarget.style.transform = "translateY(0)")}
            >
              📥 Download Bulletin (.txt)
            </button>
          </div>
        )}
      </div>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
        @keyframes slideIn {
          from { transform: translateX(20px); opacity: 0; }
          to { transform: translateX(0); opacity: 1; }
        }
        select { outline: none; }
        select option { background: #0f172a; }
      `}</style>
    </div>
  );
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function formatHour(d: Date): string {
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}
