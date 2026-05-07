#!/usr/bin/env python3
name = "12h_1d_PivotBreakout_TrendVolume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Pivot (standard) from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Pivot support/resistance levels
    s1 = pivot - range_hl
    r1 = pivot + range_hl
    
    # Align daily levels to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Volume spike detection: 4-period average (2 days of 12h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(4, 4)  # Wait for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with volume and price > previous close (momentum)
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            momentum = close[i] > close[i-1]
            
            if close[i] > s1_aligned[i] and vol_condition and momentum:
                signals[i] = 0.25
                position = 1
            # Short: price below R1 with volume and price < previous close (momentum)
            elif close[i] < r1_aligned[i] and vol_condition and not momentum:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S1 or volume drops
            if close[i] < s1_aligned[i] or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R1 or volume drops
            if close[i] > r1_aligned[i] or volume[i] < vol_ma_4[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Pivot S1/R1 breakout with volume confirmation and momentum
# - Daily Pivot S1/R1 act as key support/resistance levels from prior session
# - Breakout above S1 with volume in upward momentum = long opportunity
# - Breakdown below R1 with volume in downward momentum = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Momentum filter (price > previous close) reduces false breakouts
# - Works in both bull (buy S1 breaks in uptrend) and bear (sell R1 breaks in downtrend)
# - Exit when price returns to S1/R1 or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses actual daily Pivot levels (not weekly) for better responsiveness
# - 12h timeframe reduces noise vs lower timeframes
# - Volume confirmation reduces false breakouts
# - Momentum filter adds confluence without excessive complexity
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits
# - Focus on BTC/ETH as primary targets (avoiding SOL-only bias)