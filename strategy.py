#51
#!/usr/bin/env python3
"""
4h_PriceChannel_Volume_Spike_1dTrend_Filter
Hypothesis: Price channel breakouts (Donchian or Keltner) combined with volume spikes
and 1d trend filter work in both bull and bear markets by capturing strong momentum
moves with confirmation. Volume spikes filter out weak breakouts, while the 1d trend
filter ensures alignment with higher timeframe momentum. Target: 20-50 trades/year.
"""

name = "4h_PriceChannel_Volume_Spike_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian channel (20-period)
    lookback = 20
    highest = np.full(n, np.nan)
    lowest = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        highest[i] = np.max(high[i-lookback+1:i+1])
        lowest[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    vol_ma20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma20[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(highest[i]) or np.isnan(lowest[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + volume spike + uptrend
            if close[i] > highest[i] and volume_spike[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + volume spike + downtrend
            elif close[i] < lowest[i] and volume_spike[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price closes below Donchian low or trend reversal
            if close[i] < lowest[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price closes above Donchian high or trend reversal
            if close[i] > highest[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals