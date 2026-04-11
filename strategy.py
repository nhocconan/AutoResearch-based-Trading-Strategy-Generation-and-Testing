#!/usr/bin/env python3
# 12h_1w_camarilla_breakout_volume_v1
# Strategy: 12h Camarilla pivot breakout with 1w trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Weekly Camarilla levels from prior week act as strong support/resistance.
# Breakouts aligned with weekly trend and volume confirmation capture significant moves
# with low trade frequency (~15-30/year) to avoid fee drag. Works in both bull and bear
# markets by following the higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_breakout_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1w bar
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    rng = prev_high - prev_low
    H4 = prev_close + 1.1 * rng / 2
    L4 = prev_close - 1.1 * rng / 2
    H3 = prev_close + 1.1 * rng / 4
    L3 = prev_close - 1.1 * rng / 4
    H2 = prev_close + 1.1 * rng / 6
    L2 = prev_close - 1.1 * rng / 6
    H1 = prev_close + 1.1 * rng / 12
    L1 = prev_close - 1.1 * rng / 12
    
    # Align Camarilla levels to 12h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1w, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1w, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1w, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1w, L3)
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Breakout signals using Camarilla levels
        breakout_up = high[i] > H3_aligned[i-1]
        breakdown_down = low[i] < L3_aligned[i-1]
        
        # 1w EMA trend filter
        trend_bullish = close[i] > ema_50_1w_aligned[i]
        trend_bearish = close[i] < ema_50_1w_aligned[i]
        
        # Entry conditions
        if breakout_up and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        elif breakdown_down and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite breakout using stronger H4/L4 levels
        elif position == 1 and low[i] < L4_aligned[i-1]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and high[i] > H4_aligned[i-1]:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals