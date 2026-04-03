#!/usr/bin/env python3
"""
Experiment #074: 1h HTF 4h/1d Trend + Volume + Session Filter

HYPOTHESIS: Use 4h for trend direction (price > EMA50 for long, < for short) and 1d for volume confirmation (>1.5x 20-period average) on 1h timeframe. Enter long when 1h close > 4h EMA50 with volume spike, short when close < 4h EMA50 with volume spike. Session filter (08-20 UTC) reduces noise. Position size fixed at 0.20. Target 15-37 trades/year (60-150 total over 4 years) by using HTF for direction and 1h only for timing with strict volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_4h_ema_1d_vol_session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # === HTF: 4h data for trend direction (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate EMA(50) on 4h close
    if len(df_4h) >= 50:
        close_4h = df_4h['close'].values
        ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    else:
        ema_50_4h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # Discrete position sizing (20% of capital)
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Trend Direction from 4h EMA50 ---
        price_above_4h_ema = close[i] > ema_50_4h_aligned[i]
        price_below_4h_ema = close[i] < ema_50_4h_aligned[i]
        
        # --- Entry Logic ---
        long_condition = price_above_4h_ema and volume_spike
        short_condition = price_below_4h_ema and volume_spike
        
        if long_condition:
            signals[i] = SIZE
        elif short_condition:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals
</p>