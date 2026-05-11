#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade Camarilla pivot level breaks on 12h with daily trend filter and volume confirmation. 
Camarilla levels provide high-probability reversal/breakout zones. Only trade breaks in direction of 
daily trend to avoid counter-trend whipsaws. Volume spike confirms institutional participation. 
Works in bull/bear markets by following higher timeframe trend. Targets 15-25 trades/year.
"""

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily EMA34 Trend Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === Camarilla Levels (from previous day) ===
    # Calculate from daily OHLC
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We'll use prior day's values to avoid look-ahead
    daily_open = df_1d['open'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla for each day
    camarilla_r1 = np.zeros_like(daily_close)
    camarilla_s1 = np.zeros_like(daily_close)
    
    for i in range(len(daily_close)):
        if i == 0:
            camarilla_r1[i] = np.nan
            camarilla_s1[i] = np.nan
        else:
            prev_high = daily_high[i-1]
            prev_low = daily_low[i-1]
            prev_close = daily_close[i-1]
            camarilla_r1[i] = prev_close + (prev_high - prev_low) * 1.1 / 12
            camarilla_s1[i] = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align to 12h timeframe
    camarilla_r1_12h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_12h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === Volume Spike Filter ===
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 2.0  # 2x average volume for significance
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    
    # Start after warmup (need at least 2 days for prior day calculation)
    start_idx = 2
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_12h[i]) or np.isnan(camarilla_r1_12h[i]) or 
            np.isnan(camarilla_s1_12h[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0
            continue
        
        # Long: Break above R1 with daily uptrend and volume spike
        if (close[i] > camarilla_r1_12h[i] and 
            close[i] > ema34_12h[i] and volume_ok[i]):
            signals[i] = position_size
        # Short: Break below S1 with daily downtrend and volume spike
        elif (close[i] < camarilla_s1_12h[i] and 
              close[i] < ema34_12h[i] and volume_ok[i]):
            signals[i] = -position_size
        else:
            signals[i] = 0.0
    
    return signals