#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and 1d volume confirmation
# Uses weekly EMA50 for primary trend alignment (reduces whipsaw in ranging markets)
# Daily volume spike (>1.8x 20-bar average) confirms institutional participation
# Donchian breakout captures momentum in both bull and bear markets (trend continuation)
# Discrete sizing 0.25 to limit fee drag; target 60-120 total trades over 4 years (15-30/year)
# Proven pattern: HTF trend + LTF breakout + volume confirmation works across regimes

name = "6h_Donchian20_1wEMA50_1dVolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1w EMA50 trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume confirmation (>1.8x 20-bar average)
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.8 * vol_ma_20_1d)
    
    # Calculate 6h Donchian channels (20-bar)
    high_ma_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align HTF indicators to 6h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_spike_1d_aligned[i]) or 
            np.isnan(high_ma_20[i]) or np.isnan(low_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price > upper Donchian AND uptrend (price > weekly EMA50) AND daily volume spike
            if close[i] > high_ma_20[i] and close[i] > ema50_1w_aligned[i] and volume_spike_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < lower Donchian AND downtrend (price < weekly EMA50) AND daily volume spike
            elif close[i] < low_ma_20[i] and close[i] < ema50_1w_aligned[i] and volume_spike_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retests lower Donchian from above (trend exhaustion)
            if close[i] <= low_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retests upper Donchian from below (trend exhaustion)
            if close[i] >= high_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals