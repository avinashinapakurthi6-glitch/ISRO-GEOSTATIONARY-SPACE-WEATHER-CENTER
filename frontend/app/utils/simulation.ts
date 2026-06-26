export interface TelemetryPoint {
  time: Date;
  goes_flux: number;
  log_goes_flux: number;
  Vsw: number;
  B: number;
  Bz: number;
  N: number;
  Pdyn: number;
  bz_south: number;
  dst_proxy: number;
  vsw_lag_6h: number;
}

export type Horizon = "30m" | "6h" | "12h";
export type Quantiles = [number, number, number]; // [10th, 50th, 90th]

export interface Forecasts {
  TFT: { [key in Horizon]: Quantiles };
  LSTM: { [key in Horizon]: Quantiles };
  XGBoost: { [key in Horizon]: Quantiles };
  Persistence: { [key in Horizon]: Quantiles };
}

// Pseudo-random number generator for consistency (LCG)
function createRandom(seed: number) {
  let s = seed;
  return function () {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
}

// Exponential distribution helper
function randomExponential(randomFn: () => number, lambda: number) {
  return -Math.log(1 - randomFn()) / lambda;
}

// Normal distribution helper (Box-Muller transform)
function randomNormal(randomFn: () => number, mean: number, std: number) {
  const u1 = randomFn() || 0.0001; // Avoid 0
  const u2 = randomFn();
  const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return mean + std * z;
}

export function generateLiveStream(
  preset: string,
  vsw_val: number,
  bz_val: number,
  n_val: number
): TelemetryPoint[] {
  const random = createRandom(42);
  const pointsCount = 288;
  const now = new Date();
  
  // Initialize base arrays
  const vsw: number[] = [];
  const bz: number[] = [];
  const n: number[] = [];
  
  for (let i = 0; i < pointsCount; i++) {
    const angle1 = (i / pointsCount) * 2 * Math.PI;
    const angle2 = (i / pointsCount) * 4 * Math.PI;
    
    // Base quiet conditions
    vsw.push(350.0 + 30.0 * Math.sin(angle1) + randomNormal(random, 0, 5));
    bz.push(2.0 * Math.sin(angle2) + randomNormal(random, 0, 0.5));
    n.push(4.0 + randomExponential(random, 1.0));
  }
  
  // Ramp up parameters over the second half (index 144 to 287) if not Quiet or adjusted
  const isQuiet = preset === "Quiet conditions" && vsw_val === 350.0 && bz_val === 2.0 && n_val === 5.0;
  
  if (!isQuiet) {
    const vsw_start = vsw[144];
    const bz_start = bz[144];
    const n_start = n[144];
    
    for (let i = 144; i < pointsCount; i++) {
      const fraction = (i - 144) / 143;
      vsw[i] = vsw_start + (vsw_val - vsw_start) * fraction + randomNormal(random, 0, 8);
      bz[i] = bz_start + (bz_val - bz_start) * fraction + randomNormal(random, 0, 0.8);
      n[i] = n_start + (n_val - n_start) * fraction + randomExponential(random, 1 / 1.5);
    }
  }
  
  // Generate derived parameters
  const b: number[] = [];
  const pdyn: number[] = [];
  const bz_south: number[] = [];
  const log_goes_flux: number[] = [];
  const goes_flux: number[] = [];
  
  for (let i = 0; i < pointsCount; i++) {
    b.push(Math.abs(bz[i]) + 3.0 + randomNormal(random, 0, 0.3));
    pdyn.push(2e-6 * n[i] * Math.pow(vsw[i], 2));
    bz_south.push(Math.max(0.0, -bz[i]));
    
    const timeHour = (now.getHours() - Math.floor((pointsCount - i) * 5 / 60) + 24) % 24;
    const log_flux_base = 1.2 + 3.0 * (vsw[i] - 300.0) / 500.0;
    let log_flux = log_flux_base + 0.3 * Math.cos((2 * Math.PI * timeHour) / 24.0) + randomNormal(random, 0, 0.1);
    
    log_goes_flux.push(log_flux);
  }
  
  // Add storm dynamics physically: southward Bz causes shadowing dropout then acceleration
  let bz_south_accum = 0;
  for (let i = 144; i < pointsCount; i++) {
    bz_south_accum += bz_south[i] * 0.05;
    if (i < 180) {
      log_goes_flux[i] -= 0.1 * bz_south_accum;
    } else {
      log_goes_flux[i] += 0.08 * bz_south_accum;
    }
  }
  
  // Clamp and finalize flux
  const history: TelemetryPoint[] = [];
  let dst_accum = -10.0;
  
  for (let i = 0; i < pointsCount; i++) {
    const clamped_log = Math.max(0.5, Math.min(6.0, log_goes_flux[i]));
    const flux = Math.pow(10, clamped_log);
    
    dst_accum -= bz_south[i] * 1.2 * 0.03;
    const dst_proxy = Math.max(-180, Math.min(10, dst_accum));
    
    const time = new Date(now.getTime() - (pointsCount - i) * 5 * 60 * 1000);
    
    // Determine vsw_lag_6h (72 steps of 5 min = 6 hours)
    const lagIndex = Math.max(0, i - 72);
    const vsw_lag_6h = vsw[lagIndex];
    
    history.push({
      time,
      goes_flux: flux,
      log_goes_flux: clamped_log,
      Vsw: vsw[i],
      B: b[i],
      Bz: bz[i],
      N: n[i],
      Pdyn: pdyn[i],
      bz_south: bz_south[i],
      dst_proxy,
      vsw_lag_6h,
    });
  }
  
  return history;
}

export function computeForecasts(
  vsw_val: number,
  curr_log_flux: number
): Forecasts {
  const clamp = (val: number) => Math.max(0.5, Math.min(6.0, val));
  
  // Base TFT forecast calculation matching Python fallback
  const tft: { [key in Horizon]: Quantiles } = {
    "30m": [
      clamp(1.2 + (0.3 * (vsw_val - 350)) / 400),
      clamp(1.4 + (0.45 * (vsw_val - 350)) / 400),
      clamp(1.7 + (0.6 * (vsw_val - 350)) / 400),
    ],
    "6h": [
      clamp(1.3 + (0.4 * (vsw_val - 350)) / 400),
      clamp(1.6 + (0.6 * (vsw_val - 350)) / 400),
      clamp(2.0 + (0.8 * (vsw_val - 350)) / 400),
    ],
    "12h": [
      clamp(1.5 + (0.5 * (vsw_val - 350)) / 400),
      clamp(1.9 + (0.8 * (vsw_val - 350)) / 400),
      clamp(2.4 + (1.1 * (vsw_val - 350)) / 400),
    ],
  };
  
  // LSTM calculation (slight offset/variation from TFT)
  const lstm: { [key in Horizon]: Quantiles } = {
    "30m": [
      clamp(tft["30m"][0] - 0.05),
      clamp(tft["30m"][1] + 0.08),
      clamp(tft["30m"][2] + 0.12),
    ],
    "6h": [
      clamp(tft["6h"][0] - 0.08),
      clamp(tft["6h"][1] + 0.12),
      clamp(tft["6h"][2] + 0.18),
    ],
    "12h": [
      clamp(tft["12h"][0] - 0.12),
      clamp(tft["12h"][1] + 0.15),
      clamp(tft["12h"][2] + 0.22),
    ],
  };
  
  // XGBoost calculation (TFT - 0.15)
  const xgboost: { [key in Horizon]: Quantiles } = {
    "30m": [
      clamp(tft["30m"][0] - 0.15),
      clamp(tft["30m"][1] - 0.15),
      clamp(tft["30m"][2] - 0.15),
    ],
    "6h": [
      clamp(tft["6h"][0] - 0.15),
      clamp(tft["6h"][1] - 0.15),
      clamp(tft["6h"][2] - 0.15),
    ],
    "12h": [
      clamp(tft["12h"][0] - 0.15),
      clamp(tft["12h"][1] - 0.15),
      clamp(tft["12h"][2] - 0.15),
    ],
  };
  
  // Persistence calculation
  const persistence: { [key in Horizon]: Quantiles } = {
    "30m": [
      clamp(curr_log_flux - 0.05),
      clamp(curr_log_flux),
      clamp(curr_log_flux + 0.05),
    ],
    "6h": [
      clamp(curr_log_flux - 0.1),
      clamp(curr_log_flux),
      clamp(curr_log_flux + 0.1),
    ],
    "12h": [
      clamp(curr_log_flux - 0.15),
      clamp(curr_log_flux),
      clamp(curr_log_flux + 0.15),
    ],
  };
  
  return {
    TFT: tft,
    LSTM: lstm,
    XGBoost: xgboost,
    Persistence: persistence,
  };
}
