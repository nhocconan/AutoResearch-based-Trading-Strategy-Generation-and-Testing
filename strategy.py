#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Donchian(20) upper AND 1d ADX > 25 (trending) AND volume > 1.5x 20 EMA
# Short when price breaks below Donchian(20) lower AND 1d ADX > 25 (trending) AND volume > 1.5x 20 EMA
# Exit on opposite Donchian break or ADX < 20 (range regime)
# Uses discrete sizing (0.25) to limit fee drag. Target: 20-50 trades/year per symbol.
# Works in bull markets via longs in uptrends and bear markets via shorts in downtrends.
# Uses 1d for HTF trend to avoid counter-trend trades and 4h for entry timing.

name = "4h_Donchian20_1dADX_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get daily OHLC arrays
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels for 4h (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d True Range for ADX
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate 1d ADX (14-period)
    period = 14
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values
    atr_1d = pd.Series(tr_1d).ewm(span=period, adjust=False, min_periods=period).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False, min_periods=period).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirm = volume > (vol_ema_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND 1d ADX > 25 (trending) AND volume confirm
            if (close[i] > high_20[i] and 
                adx_1d_aligned[i] > 25 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND 1d ADX > 25 (trending) AND volume confirm
            elif (close[i] < low_20[i] and 
                  adx_1d_aligned[i] > 25 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower OR 1d ADX < 20 (range regime)
            if (close[i] < low_20[i] or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper OR 1d ADX < 20 (range regime)
            if (close[i] > high_20[i] or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals