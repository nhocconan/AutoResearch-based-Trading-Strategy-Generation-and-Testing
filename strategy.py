#!/usr/bin/env python3
name = "1h_4h_1d_Camarilla_S1R1_Breakout_VolumeTrend"
timeframe = "1h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    s1 = prev_close - (range_hl * 1.08 / 2)
    r1 = prev_close + (range_hl * 1.08 / 2)
    s2 = prev_close - (range_hl * 1.16 / 2)
    r2 = prev_close + (range_hl * 1.16 / 2)
    s3 = prev_close - (range_hl * 1.26 / 4)
    r3 = prev_close + (range_hl * 1.26 / 4)
    
    # Align daily levels to 1h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # Daily trend filter: EMA(34) on daily close
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 1)  # Wait for EMA
    
    for i in range(start_idx, n):
        # Skip if not in session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above S1 with daily uptrend
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            if close[i] > s1_aligned[i] and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: price below R1 with daily downtrend
            elif close[i] < r1_aligned[i] and not uptrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price back below S1 or trend changes
            if close[i] < s1_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above R1 or trend changes
            if close[i] > r1_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Camarilla S1/R1 breakout with daily trend filter and session filter
# - Uses daily Camarilla S1/R1 as key support/resistance levels from higher timeframe
# - Enters long when price breaks above S1 during daily uptrend (08:00-20:00 UTC)
# - Enters short when price breaks below R1 during daily downtrend (08:00-20:00 UTC)
# - Exits when price returns to S1/R1 or daily trend changes
# - Session filter reduces noise trades by focusing on active market hours
# - Position size 0.20 manages risk while allowing meaningful returns
# - Designed to work in both bull and bear markets via daily trend filter
# - Target: 15-35 trades/year to avoid excessive fee drag on 1h timeframe