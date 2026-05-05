#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume confirmation and 1d ADX regime filter
# Long when: price breaks above Donchian upper (20) AND volume > 1.5x 24-period MA AND 1d ADX > 25 (trending)
# Short when: price breaks below Donchian lower (20) AND volume > 1.5x 24-period MA AND 1d ADX > 25 (trending)
# Exit when: price returns to Donchian midpoint OR ADX < 20 (range regime)
# Uses Donchian for breakout structure, volume for conviction, 1d ADX for regime filter to avoid whipsaws
# Timeframe: 12h, HTF: 1d. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Donchian20_Volume_ADXRegime"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 12h using 24-period MA (equivalent to 1d lookback)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_filter = volume > (1.5 * vol_ma_24)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Donchian channels on 12h (20-period)
    if len(high) >= 20 and len(low) >= 20:
        donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donch_mid = (donch_high + donch_low) / 2
    else:
        donch_high = np.full(n, np.nan)
        donch_low = np.full(n, np.nan)
        donch_mid = np.full(n, np.nan)
    
    # Get 1d data ONCE before loop for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need enough data for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on 1d timeframe
    if len(high_1d) >= 14 and len(low_1d) >= 14 and len(close_1d) >= 14:
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # prepend NaN for first element
        
        # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
        up_move = high_1d[1:] - high_1d[:-1]
        down_move = low_1d[:-1] - low_1d[1:]
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/14)
        def WilderSmoothing(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[1:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1] / period) + (data[i] / period)
                else:
                    result[i] = np.nan
            return result
        
        atr = WilderSmoothing(tr, 14)
        plus_di = 100 * WilderSmoothing(plus_dm, 14) / atr
        minus_di = 100 * WilderSmoothing(minus_dm, 14) / atr
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = WilderSmoothing(dx, 14)
        
        # ADX regime filters
        adx_strong_trend = adx > 25  # trending regime
        adx_weak_trend = adx < 20    # range regime (exit condition)
    else:
        adx_strong_trend = np.full(len(close_1d), False)
        adx_weak_trend = np.full(len(close_1d), False)
    
    # Align 1d ADX regime to 12h timeframe
    adx_strong_trend_aligned = align_htf_to_ltf(prices, df_1d, adx_strong_trend.astype(float))
    adx_weak_trend_aligned = align_htf_to_ltf(prices, df_1d, adx_weak_trend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(adx_strong_trend_aligned[i]) or 
            np.isnan(adx_weak_trend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper + volume filter + strong trend regime
            if (close[i] > donch_high[i] and 
                volume_filter[i] and 
                adx_strong_trend_aligned[i] == 1.0):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower + volume filter + strong trend regime
            elif (close[i] < donch_low[i] and 
                  volume_filter[i] and 
                  adx_strong_trend_aligned[i] == 1.0):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Donchian midpoint OR weak trend regime (range)
            if (close[i] <= donch_mid[i] or adx_weak_trend_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Donchian midpoint OR weak trend regime (range)
            if (close[i] >= donch_mid[i] or adx_weak_trend_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals