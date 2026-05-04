#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction + volume confirmation
# Long when price breaks above upper Donchian channel AND weekly bullish pivot (price > weekly pivot) AND volume > 1.5x 20-period volume EMA
# Short when price breaks below lower Donchian channel AND weekly bearish pivot (price < weekly pivot) AND volume > 1.5x 20-period volume EMA
# Uses weekly pivot points (calculated from prior week OHLC) for structural bias, reducing whipsaw in ranging markets.
# Volume spike filter confirms breakout strength. Target: 12-37 trades/year on 6h.

name = "6h_Donchian20_WeeklyPivot_VolumeSpike"
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
    
    # Get 1w data for weekly pivot calculation - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points: P = (H+L+C)/3, R1 = 2*P - L, S1 = 2*P - H
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot (using prior week's values to avoid look-ahead)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Bullish bias: price above pivot, Bearish bias: price below pivot
    bullish_bias_1w = close_1w > pivot_1w
    bearish_bias_1w = close_1w < pivot_1w
    
    # Align weekly bias to 6h timeframe (completed weekly bar only)
    bullish_bias_aligned = align_htf_to_ltf(prices, df_1w, bullish_bias_1w.astype(float))
    bearish_bias_aligned = align_htf_to_ltf(prices, df_1w, bearish_bias_1w.astype(float))
    
    # Calculate Donchian channels (20-period) from 6h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bullish_bias_aligned[i]) or np.isnan(bearish_bias_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND weekly bullish bias AND volume spike
            if (close[i] > donchian_upper[i] and 
                bullish_bias_aligned[i] > 0.5 and  # Weekly bullish bias
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND weekly bearish bias AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  bearish_bias_aligned[i] > 0.5 and  # Weekly bearish bias
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower OR weekly bias turns bearish
            if (close[i] < donchian_lower[i] or 
                bearish_bias_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper OR weekly bias turns bullish
            if (close[i] > donchian_upper[i] or 
                bullish_bias_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals