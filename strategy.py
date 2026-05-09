#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Uses Donchian channels for breakout detection, ADX(14) > 25 for trending markets, and volume > 1.5x 20-period average
# Only takes longs when price breaks above upper Donchian band AND ADX > 25 AND volume spike
# Only takes shorts when price breaks below lower Donchian band AND ADX > 25 AND volume spike
# Exits when price crosses the Donchian midline (average of upper/lower bands) or ADX falls below 20
# Target: 20-40 trades per year with position size 0.25 for balanced risk/return in both bull and bear markets

name = "4h_Donchian_20_1dADX_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX(14) on daily timeframe
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: smoothed = previous_smoothed - (previous_smoothed/period) + current_value
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1]/period) + data[i]
            else:
                result[i] = np.nan
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align daily ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian(20) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_20
    donchian_lower = low_20
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Volume spike: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + ADX > 25 + volume spike
            if (close[i] > donchian_upper[i] and 
                adx_aligned[i] > 25 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian + ADX > 25 + volume spike
            elif (close[i] < donchian_lower[i] and 
                  adx_aligned[i] > 25 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below middle Donchian OR ADX < 20 (trend weakening)
            if (close[i] < donchian_middle[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above middle Donchian OR ADX < 20 (trend weakening)
            if (close[i] > donchian_middle[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals