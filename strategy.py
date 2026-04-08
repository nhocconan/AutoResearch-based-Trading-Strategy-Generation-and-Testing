#!/usr/bin/env python3
# 4h_donchian_breakout_1d_trend_volume_v3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA200 trend filter and volume confirmation.
# Enters long when price breaks above upper Donchian channel and closes above EMA200 with volume > 1.5x average.
# Enters short when price breaks below lower Donchian channel and closes below EMA200 with volume > 1.5x average.
# Exits when price returns to the middle of the Donchian channel.
# Designed for ~30 trades/year on 4h to avoid fee drag. Works in bull/bear via trend-following with strong filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v3"
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
    
    # 1-day data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 4-period average volume for confirmation
    vol_ma = np.zeros(n)
    for i in range(4, n):
        vol_ma[i] = np.mean(volume[i-4:i])
    
    # Donchian channel (20-period) on 4h data
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper and lower bands
    upper = np.zeros(len(high_4h))
    lower = np.zeros(len(low_4h))
    for i in range(20, len(high_4h)):
        upper[i] = np.max(high_4h[i-20:i])
        lower[i] = np.min(low_4h[i-20:i])
    
    # Align Donchian levels to 4h timeframe (already aligned via get_htf_data)
    # No need to align as we're using 4h data directly on 4h timeframe
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(200, 20)  # Ensure EMA200 and Donchian are ready
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Current 4h bar index (since we're on 4h timeframe)
        idx_4h = i
        
        # Volume confirmation: current volume > 1.5x 4-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if i >= 4 else False
        
        # Trend filter
        price_above_ema = close[i] > ema200_1d_aligned[i]
        price_below_ema = close[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to middle of Donchian channel
            mid = (upper[idx_4h] + lower[idx_4h]) / 2
            if close[i] <= mid:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to middle of Donchian channel
            mid = (upper[idx_4h] + lower[idx_4h]) / 2
            if close[i] >= mid:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian with volume and above EMA200
            if close[i] > upper[idx_4h] and vol_confirm and price_above_ema:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian with volume and below EMA200
            elif close[i] < lower[idx_4h] and vol_confirm and price_below_ema:
                position = -1
                signals[i] = -0.25
    
    return signals