#!/usr/bin/env python3
name = "6h_Donchian_20_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly Pivot Point (PP) and support/resistance
    prev_weekly_high = df_1w['high'].shift(1).values
    prev_weekly_low = df_1w['low'].shift(1).values
    prev_weekly_close = df_1w['close'].shift(1).values
    
    weekly_pp = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3
    weekly_r1 = 2 * weekly_pp - prev_weekly_low
    weekly_s1 = 2 * weekly_pp - prev_weekly_high
    
    # Align weekly pivot levels to 6h
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    
    # Donchian channel (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 1.8 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian
    
    for i in range(start_idx, n):
        if np.isnan(weekly_pp_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above Donchian high with volume spike and price above weekly pivot
            if high[i] > donchian_high[i] and vol_spike[i] and close[i] > weekly_pp_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume spike and price below weekly pivot
            elif low[i] < donchian_low[i] and vol_spike[i] and close[i] < weekly_pp_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price below weekly S1 or Donchian low breaks
            if close[i] < weekly_s1_aligned[i] or low[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price above weekly R1 or Donchian high breaks
            if close[i] > weekly_r1_aligned[i] or high[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout on 6h with weekly pivot direction filter and volume confirmation.
# Long when price breaks above Donchian high with volume spike and price above weekly pivot (bullish bias).
# Short when price breaks below Donchian low with volume spike and price below weekly pivot (bearish bias).
# Weekly pivot provides longer-term directional bias to avoid counter-trend breakouts.
# Volume spike (>1.8x average) ensures conviction behind the breakout.
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in bull markets (breakouts in uptrend) and bear markets (breakdowns in downtrend).