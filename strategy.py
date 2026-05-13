#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h Donchian(20) breakout direction filtered by 1d EMA50 trend, with volume confirmation (>1.5x 20-bar avg) and session filter (08-20 UTC).
# Uses 1d EMA50 for primary trend alignment (HTF), 4h Donchian channels for structural breakout signals, and volume to avoid false breakouts.
# Session filter reduces noise during low-liquidity hours. Targets 60-150 total trades over 4 years to minimize fee drag.
# Works in bull markets by following uptrend breaks, and in bear markets by shorting downtrend breaks — both require volume confirmation.

name = "1h_Donchian20_Breakout_1dEMA50_VolumeConfirm_Session_v1"
timeframe = "1h"
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
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Donchian channels (20-period) for breakout signals
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    highest_high_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().shift(1).values
    highest_high_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)
    lowest_low_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Session filter: only trade between 08:00 and 20:00 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high_4h_aligned[i]) or 
            np.isnan(lowest_low_4h_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 4h Donchian upper channel, close > 1d EMA50, volume spike (>1.5x avg)
            if (high[i] > highest_high_4h_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below 4h Donchian lower channel, close < 1d EMA50, volume spike (>1.5x avg)
            elif (low[i] < lowest_low_4h_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close if price breaks below 4h Donchian lower channel or volume drops significantly
            if (low[i] < lowest_low_4h_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Close if price breaks above 4h Donchian upper channel or volume drops significantly
            if (high[i] > highest_high_4h_aligned[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals