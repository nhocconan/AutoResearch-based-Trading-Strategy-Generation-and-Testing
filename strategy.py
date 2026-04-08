#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v2
# Hypothesis: 4h Donchian breakout with 1d EMA200 trend filter and volume confirmation.
# Enters long when price breaks above Donchian(20) high, price > 1d EMA200, and volume > 1.5x 20-period average.
# Enters short when price breaks below Donchian(20) low, price < 1d EMA200, and volume > 1.5x 20-period average.
# Exits when price crosses back across Donchian middle (10-period average) or volume drops below average.
# Designed for ~25-35 trades/year on 4h to avoid fee drag. Works in bull/bear via trend-following with volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # Donchian middle (10-period average of high/low)
    highest_high_10 = np.full(n, np.nan)
    lowest_low_10 = np.full(n, np.nan)
    for i in range(10, n):
        highest_high_10[i] = np.max(high[i-10:i])
        lowest_low_10[i] = np.min(low[i-10:i])
    donchian_middle = (highest_high_10 + lowest_low_10) / 2
    
    # 1-day EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_threshold = vol_ma20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, 200)  # Ensure Donchian and EMA200 are ready
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > volume_threshold[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian middle OR volume drops below average
            if close[i] < donchian_middle[i] or not vol_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian middle OR volume drops below average
            if close[i] > donchian_middle[i] or not vol_ok:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high, price > 1d EMA200, volume confirmation
            if close[i] > highest_high[i] and close[i] > ema200_1d_aligned[i] and vol_ok:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low, price < 1d EMA200, volume confirmation
            elif close[i] < lowest_low[i] and close[i] < ema200_1d_aligned[i] and vol_ok:
                position = -1
                signals[i] = -0.25
    
    return signals