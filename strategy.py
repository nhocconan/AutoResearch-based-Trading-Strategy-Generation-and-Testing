#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian(20) breakout with 12-hour EMA50 trend filter and volume confirmation.
- Long when price breaks above 4h Donchian high(20) with volume spike and above 12h EMA50
- Short when price breaks below 4h Donchian low(20) with volume spike and below 12h EMA50
- Exit when price crosses the opposite Donchian band or trend reverses
- Volume confirmation: current volume > 1.8x 20-period average
- Target: 20-40 trades/year (80-160 over 4 years) to minimize fee drag
- Works in bull/bear via trend filter and breakout structure
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Volume spike: current volume > 1.8 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough data for Donchian and EMA
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high with volume spike and above 12h EMA50
            if (close[i] > donch_high[i] and volume_spike[i] and 
                close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low with volume spike and below 12h EMA50
            elif (close[i] < donch_low[i] and volume_spike[i] and 
                  close[i] < ema50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price crosses below Donchian low OR below 12h EMA50 (trend change)
            if (close[i] < donch_low[i] or close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian high OR above 12h EMA50 (trend change)
            if (close[i] > donch_high[i] or close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0