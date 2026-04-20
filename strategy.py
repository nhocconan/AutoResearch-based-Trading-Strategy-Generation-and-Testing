#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d volume spike and ADX filter
# - Long when price breaks above 12h Donchian upper channel (20) + 1d volume > 1.5x 20-day average + ADX(14) > 20
# - Short when price breaks below 12h Donchian lower channel (20) + 1d volume > 1.5x 20-day average + ADX(14) > 20
# - Exit when price crosses back through the opposite Donchian channel (exit at lower band for long, upper band for short)
# - Uses Donchian channels for trend structure, volume for conviction, ADX to avoid choppy markets
# - Designed for 12h timeframe with selective entries to target 12-37 trades per year

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for volume average
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    
    # Calculate 20-day average volume on 1d
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate ADX(14) on 12h timeframe
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]  # first period
    tr2[0] = high[0] - close[0]
    tr3[0] = close[0] - low[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI and DX
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period) on 12h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian/ADX warmup
        # Skip if NaN in indicators
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = vol_1d[i]  # current 1d volume
        
        if position == 0:
            # Long entry: price breaks above upper Donchian + volume spike + ADX > 20
            if price > highest_high[i] and vol > 1.5 * vol_ma_1d_aligned[i] and adx[i] > 20:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian + volume spike + ADX > 20
            elif price < lowest_low[i] and vol > 1.5 * vol_ma_1d_aligned[i] and adx[i] > 20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower Donchian channel
            if price < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper Donchian channel
            if price > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeSpike_ADXFilter"
timeframe = "12h"
leverage = 1.0