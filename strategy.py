#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume spike and 1d ADX trend filter
# Long when: price breaks above Donchian upper band (20-period high), volume > 1.5x 20-period average, and 1d ADX > 25
# Short when: price breaks below Donchian lower band (20-period low), volume > 1.5x 20-period average, and 1d ADX > 25
# Exit when price returns to the Donchian midpoint (mean reversion) or opposite breakout
# Uses Donchian channels from 4h for structure, effective in both bull (breakout continuation) and bear (mean reversion via exits) markets.
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Donchian20_Breakout_1dADX25_VolumeSpike"
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
    open_price = prices['open'].values
    
    # Calculate volume confirmation on 4h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian channels on 4h (20-period)
    if len(high) >= 20:
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_mid = (donchian_upper + donchian_lower) / 2
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Get 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # Align with original index
        
        # Directional Movement
        up_move = np.diff(high_1d)
        down_move = -np.diff(low_1d)
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        # Smoothed values (Wilder's smoothing)
        def wilders_smoothing(values, period):
            result = np.full_like(values, np.nan)
            if len(values) < period:
                return result
            # First value is simple average
            result[period-1] = np.nansum(values[1:period])  # Skip index 0 (no prior period)
            for i in range(period, len(values)):
                result[i] = result[i-1] - (result[i-1] / period) + values[i]
            return result
        
        atr = wilders_smoothing(tr, 14)
        plus_di = 100 * wilders_smoothing(plus_dm, 14) / atr
        minus_di = 100 * wilders_smoothing(minus_dm, 14) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = wilders_smoothing(dx, 14)
    else:
        adx = np.full(len(close_1d), np.nan)
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(adx_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, volume filter, and 1d ADX > 25
            if (close[i] > donchian_upper[i] and 
                open_price[i] <= donchian_upper[i] and  # Ensure breakout happens on this bar
                volume_filter[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower, volume filter, and 1d ADX > 25
            elif (close[i] < donchian_lower[i] and 
                  open_price[i] >= donchian_lower[i] and  # Ensure breakdown happens on this bar
                  volume_filter[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below Donchian midpoint (mean reversion) or breaks below lower (reversal)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above Donchian midpoint (mean reversion) or breaks above upper (reversal)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals