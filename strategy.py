#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla R1/S1 breakout on 12h with 1d volume spike and 1d EMA34 trend filter.
Only trade breakouts in the direction of the 1d trend to avoid counter-trend whipsaws.
Volume spike confirms institutional interest. Designed for lower trade frequency on 12h timeframe
to minimize fee drag while capturing sustained moves in both bull and bear markets.
"""

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
    
    # Calculate 1d Camarilla pivot levels (R1/S1 only)
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
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume spike: current volume > 2.0 * 20-period average
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = df_1d['volume'].values > (2.0 * vol_avg_1d)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for EMA34 and volume average
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_trend = ema_34_aligned[i]
        vol_spike = bool(volume_spike_aligned[i])
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for entry in direction of 1d trend with volume confirmation
            if close_val > ema_trend:  # Uptrend
                long_entry = (close_val > R1_aligned[i]) and vol_spike
                if long_entry:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            else:  # Downtrend
                short_entry = (close_val < S1_aligned[i]) and vol_spike
                if short_entry:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Long - exit if price breaks below S1 (trend failure) or reverses below EMA
            if close_val < S1_aligned[i] or close_val < ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit if price breaks above R1 (trend failure) or reverses above EMA
            if close_val > R1_aligned[i] or close_val > ema_trend:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0