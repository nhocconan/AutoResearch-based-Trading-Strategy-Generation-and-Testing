#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly pivot reversion with volume confirmation and weekly trend filter.
# Long when price touches S1/S2 pivot from previous week with weekly uptrend and volume spike.
# Short when price touches R1/R2 pivot with weekly downtrend and volume spike.
# Uses mean-reversion at weekly pivot levels, which works in both bull and bear markets.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation and trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly pivot points and support/resistance levels
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    pivot = (high_weekly + low_weekly + close_weekly) / 3.0
    r1 = 2 * pivot - low_weekly
    s1 = 2 * pivot - high_weekly
    r2 = pivot + (high_weekly - low_weekly)
    s2 = pivot - (high_weekly - low_weekly)
    
    # Align weekly levels to daily timeframe (wait for weekly bar to close)
    r1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    r2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s1_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    s2_weekly_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Weekly trend filter: 20-period EMA on weekly close
    ema20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_weekly_aligned[i]) or np.isnan(r2_weekly_aligned[i]) or
            np.isnan(s1_weekly_aligned[i]) or np.isnan(s2_weekly_aligned[i]) or
            np.isnan(ema20_weekly_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price touches S1 or S2 AND weekly uptrend AND volume spike
        long_condition = ((abs(close[i] - s1_weekly_aligned[i]) < 0.001 * close[i] or
                          abs(close[i] - s2_weekly_aligned[i]) < 0.001 * close[i]) and
                         close[i] > ema20_weekly_aligned[i] and
                         volume_filter[i])
        
        # Short conditions: price touches R1 or R2 AND weekly downtrend AND volume spike
        short_condition = ((abs(close[i] - r1_weekly_aligned[i]) < 0.001 * close[i] or
                           abs(close[i] - r2_weekly_aligned[i]) < 0.001 * close[i]) and
                          close[i] < ema20_weekly_aligned[i] and
                          volume_filter[i])
        
        if long_condition:
            signals[i] = 0.25
            position = 1
        elif short_condition:
            signals[i] = -0.25
            position = -1
        else:
            # Hold current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyPivotReversion_Volume"
timeframe = "1d"
leverage = 1.0