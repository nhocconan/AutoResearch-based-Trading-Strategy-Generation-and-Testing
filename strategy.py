#!/usr/bin/env python3
name = "4h_Keltner_Channel_Pullback_1dTrend_Volume"
timeframe = "4h"
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
    
    # === 1d EMA20 (Keltner middle) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # === ATR(10) for Keltner channels ===
    atr_1d = np.zeros_like(close_1d)
    tr_1d = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner channels: upper = EMA20 + 2*ATR, lower = EMA20 - 2*ATR
    keltner_upper_1d = ema20_1d + 2.0 * atr_1d
    keltner_lower_1d = ema20_1d - 2.0 * atr_1d
    
    # === 1d Volume spike filter ===
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * vol_avg_1d)
    
    # Align to 4h
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    keltner_upper_1d_aligned = align_htf_to_ltf(prices, df_1d, keltner_upper_1d)
    keltner_lower_1d_aligned = align_htf_to_ltf(prices, df_1d, keltner_lower_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_1d_aligned[i]) or 
            np.isnan(keltner_upper_1d_aligned[i]) or
            np.isnan(keltner_lower_1d_aligned[i]) or
            np.isnan(vol_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Pullback to lower Keltner + above EMA20 + volume spike
            if (low[i] <= keltner_lower_1d_aligned[i] and
                close[i] > ema20_1d_aligned[i] and
                vol_spike_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Pullback to upper Keltner + below EMA20 + volume spike
            elif (high[i] >= keltner_upper_1d_aligned[i] and
                  close[i] < ema20_1d_aligned[i] and
                  vol_spike_1d_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below EMA20 or pullback to upper band
            if close[i] < ema20_1d_aligned[i] or high[i] >= keltner_upper_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above EMA20 or pullback to lower band
            if close[i] > ema20_1d_aligned[i] or low[i] <= keltner_lower_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals