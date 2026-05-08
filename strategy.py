#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Weekly_Pivot_Trend_Continuation_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data once
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly pivot levels (from previous week) ===
    prev_high_1w = np.roll(df_1w['high'].values, 1)
    prev_low_1w = np.roll(df_1w['low'].values, 1)
    prev_close_1w = np.roll(df_1w['close'].values, 1)
    prev_high_1w[0] = df_1w['high'].values[0]
    prev_low_1w[0] = df_1w['low'].values[0]
    prev_close_1w[0] = df_1w['close'].values[0]
    
    pivot_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3.0
    range_1w = prev_high_1w - prev_low_1w
    
    # Weekly support/resistance levels (standard pivot)
    r1_1w = pivot_1w + (range_1w * 1.0 / 3)
    s1_1w = pivot_1w - (range_1w * 1.0 / 3)
    r2_1w = pivot_1w + (range_1w * 2.0 / 3)
    s2_1w = pivot_1w - (range_1w * 2.0 / 3)
    
    # Align weekly levels to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_6h = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_6h = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_6h = align_htf_to_ltf(prices, df_1w, s2_1w)
    
    # === 6h 20-period EMA for trend filter ===
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === 6h volume filter: current volume > 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or np.isnan(r2_6h[i]) or np.isnan(s2_6h[i]) or
            np.isnan(ema20[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R2 with trend and volume confirmation
            long_cond = (close[i] > r2_6h[i] and 
                        close[i] > ema20[i] and
                        volume[i] > vol_ma20[i])
            
            # Short: price breaks below S2 with trend and volume confirmation
            short_cond = (close[i] < s2_6h[i] and 
                         close[i] < ema20[i] and
                         volume[i] > vol_ma20[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below R1 (mean reversion) or trend fails
            exit_cond = (close[i] < r1_6h[i] or 
                        close[i] < ema20[i])
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above S1 (mean reversion) or trend fails
            exit_cond = (close[i] > s1_6h[i] or 
                        close[i] > ema20[i])
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot-based trend continuation strategy for 6H timeframe.
# Enters long when price breaks above weekly R2 with EMA20 uptrend and volume confirmation.
# Enters short when price breaks below weekly S2 with EMA20 downtrend and volume confirmation.
# Exits when price returns to weekly R1/S1 (mean reversion) or trend fails (price crosses EMA20).
# Uses weekly pivots as institutional reference points - effective in both bull and bear markets.
# Targets 50-150 trades over 4 years (12-37/year) to minimize fee drag.
# Uses discrete sizing (0.25) to reduce churn. Weekly timeframe avoids noise from lower TFs.