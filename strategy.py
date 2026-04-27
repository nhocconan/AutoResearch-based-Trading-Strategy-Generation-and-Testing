#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike
Hypothesis: On 1h timeframe, trade Camarilla R1/S1 breakouts only when aligned with 4h trend (price > EMA50) and confirmed by 1d volume spike. 
In bear markets, 4h EMA50 filter prevents whipsaw trades; volume spike ensures institutional participation. 
Session filter (08-20 UTC) reduces noise. Target 15-35 trades/year by requiring confluence of 3 filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d Camarilla R1/S1 levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    PP = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    R1 = PP + range_1d * 1.0 / 4.0
    S1 = PP - range_1d * 1.0 / 4.0
    
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d volume spike: current volume > 2.0 * 20-day average
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (2.0 * vol_avg_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # 20% position size
    
    # Warmup: need enough for all indicators
    start_idx = max(100, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        
        if position == 0:
            # Flat - look for entry: breakout with 4h trend and volume spike
            long_entry = (close_val > R1_aligned[i]) and (close_val > ema_4h_aligned[i]) and volume_spike_1d_aligned[i]
            short_entry = (close_val < S1_aligned[i]) and (close_val < ema_4h_aligned[i]) and volume_spike_1d_aligned[i]
            
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Long - exit: close below S1 or 4h EMA50 (trend change)
            if close_val < S1_aligned[i] or close_val < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit: close above R1 or 4h EMA50 (trend change)
            if close_val > R1_aligned[i] or close_val > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeSpike"
timeframe = "1h"
leverage = 1.0