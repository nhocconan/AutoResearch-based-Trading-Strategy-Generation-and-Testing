#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_200EMA_Filter_VolumeSpike_4hTrend
Hypothesis: Breakout above/below 4h Donchian(20) channel with trend filter (price > 200 EMA),
volume confirmation (>1.5x 20-period volume average), and exit at opposite Donchian level.
Designed for low-frequency, high-conviction trades in both bull and bear markets by requiring
multiple confluence factors. Uses 4h timeframe with 1h EMA for entry timing precision.
"""

name = "4h_Donchian20_Breakout_200EMA_Filter_VolumeSpike_4hTrend"
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
    
    # Calculate 4h Donchian(20) channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 200 EMA for trend filter (using 1h data for finer resolution)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 200:
        return np.zeros(n)
    
    close_1h = df_1h['close'].values
    ema_200_1h = pd.Series(close_1h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_200_1h)
    
    # Volume spike condition: current volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 8  # Prevent overtrading
    
    start_idx = max(20, 200)  # Warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_200_1h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Donchian breakout above, price > 200 EMA, volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_200_1h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Donchian breakdown below, price < 200 EMA, volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_200_1h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit long: price crosses below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals