#!/usr/bin/env python3
"""
1d Donchian Breakout + Weekly Volume Spike + KAMA Trend Filter
Hypothesis: Donchian breakouts capture strong trends; weekly volume spike confirms institutional interest;
KAMA(30) filters false breakouts in ranging markets. Designed for low trade frequency (<20/year)
to minimize fee decay while capturing sustained moves in bull/bear markets.
"""
name = "1d_Donchian_Volume_KAMA"
timeframe = "1d"
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
    
    # === Weekly KAMA Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility = np.abs(np.diff(close_1w))
    er = np.zeros(len(close_1w))
    for i in range(1, len(close_1w)):
        if np.sum(volatility[max(0, i-9):i+1]) > 0:
            er[i] = change[i] / np.sum(volatility[max(0, i-9):i+1])
        else:
            er[i] = 0
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_1w = np.zeros(len(close_1w))
    kama_1w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_1w[i] = kama_1w[i-1] + sc[i] * (close_1w[i] - kama_1w[i-1])
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # === Daily Donchian Channels (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Weekly Volume Spike (20) ===
    vol_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = vol_1w > (vol_ma_1w * 2.0)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure all indicators ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_spike_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + weekly KAMA uptrend + weekly volume spike
            if (close[i] > highest_high[i] and 
                close[i] > kama_1w_aligned[i] and 
                vol_spike_1w_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + weekly KAMA downtrend + weekly volume spike
            elif (close[i] < lowest_low[i] and 
                  close[i] < kama_1w_aligned[i] and 
                  vol_spike_1w_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Price closes below weekly KAMA
            if close[i] < kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above weekly KAMA
            if close[i] > kama_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals