#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
- Long: Close breaks above R1 (Camarilla resistance) + price > 1d EMA34 (uptrend) + volume > 1.5x 20-period avg
- Short: Close breaks below S1 (Camarilla support) + price < 1d EMA34 (downtrend) + volume > 1.5x 20-period avg
- Exit: Close crosses 1d EMA34 in opposite direction OR momentum exhaustion (volume < 0.5x avg for 2 consecutive bars)
- Uses proven Camarilla structure from 1d pivots, effective in both bull (breakouts) and bear (mean reversion at levels)
- Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag on 4h timeframe
"""

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
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for pivot calculation and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (based on previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = Close + (High - Low) * 1.1/12, S1 = Close - (High - Low) * 1.1/12
    camarilla_range = (high_1d - low_1d) * 1.1 / 12
    r1 = close_1d + camarilla_range
    s1 = close_1d - camarilla_range
    
    # Align 1d levels to 4h timeframe (only update when new 1d bar completes)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34)  # Need 20 for volume MA, 34 for 1d EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        # Volume exhaustion filter (< 0.5x average for 2 consecutive bars)
        if i >= 1:
            volume_exhaustion = (volume[i] < 0.5 * vol_ma[i]) and (volume[i-1] < 0.5 * vol_ma[i-1])
        else:
            volume_exhaustion = False
        
        if position == 0:
            # Long: Close breaks above R1 + price > 1d EMA34 (uptrend) + volume spike
            if volume_spike and close[i] > r1_aligned[i] and close[i] > ema_34_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1 + price < 1d EMA34 (downtrend) + volume spike
            elif volume_spike and close[i] < s1_aligned[i] and close[i] < ema_34_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close crosses below 1d EMA34 (trend break) OR volume exhaustion
            if close[i] < ema_34_aligned[i] or volume_exhaustion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Close crosses above 1d EMA34 (trend break) OR volume exhaustion
            if close[i] > ema_34_aligned[i] or volume_exhaustion:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
leverage = 1.0