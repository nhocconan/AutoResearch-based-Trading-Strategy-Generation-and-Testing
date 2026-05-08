#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R1/S1) breakout with 1d volume spike and 1w EMA50 trend filter.
# Long when price breaks above R1, volume > 1.5x 20-period average, and price > 1w EMA50.
# Short when price breaks below S1, volume > 1.5x 20-period average, and price < 1w EMA50.
# Exit when price crosses back below R1 (for long) or above S1 (for short).
# Camarilla levels provide precise intraday support/resistance; volume confirms breakout strength.
# Weekly EMA50 filters for higher timeframe trend to avoid counter-trend trades.
# Target: 75-200 total trades over 4 years (19-50/year) for low fee drift.

name = "4h_Camarilla_R1_S1_Breakout_1dVolume_1wEMA50"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (using previous day's HLC)
    # For each 4h bar, we need the previous day's H, L, C
    # We'll shift the 1d data by 1 to get previous day's values
    prev_day_high = df_1d['high'].shift(1).values
    prev_day_low = df_1d['low'].shift(1).values
    prev_day_close = df_1d['close'].shift(1).values
    
    # Camarilla calculations
    R1 = prev_day_close + (prev_day_high - prev_day_low) * 1.0833 / 12
    S1 = prev_day_close - (prev_day_high - prev_day_low) * 1.0833 / 12
    
    # Align Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d volume filter: current 1d volume > 1.5x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (1.5 * vol_ma20_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1, volume spike, above 1w EMA50
            long_cond = (close[i] > R1_aligned[i]) and (close[i-1] <= R1_aligned[i-1]) and volume_spike_1d_aligned[i] and (close[i] > ema50_1w_aligned[i])
            # Short conditions: price breaks below S1, volume spike, below 1w EMA50
            short_cond = (close[i] < S1_aligned[i]) and (close[i-1] >= S1_aligned[i-1]) and volume_spike_1d_aligned[i] and (close[i] < ema50_1w_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below R1
            if close[i] < R1_aligned[i] and close[i-1] >= R1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above S1
            if close[i] > S1_aligned[i] and close[i-1] <= S1_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals