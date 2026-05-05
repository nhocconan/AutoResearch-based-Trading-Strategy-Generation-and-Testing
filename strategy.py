#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + volume confirmation + 1d ADX regime filter
# Long when price breaks above Donchian upper band AND volume > 1.5x 20-bar average AND 1d ADX > 25
# Short when price breaks below Donchian lower band AND volume > 1.5x 20-bar average AND 1d ADX > 25
# Exit when price crosses Donchian middle band (20-bar average) OR opposite Donchian band is touched
# Uses 4h primary timeframe with 1d HTF for ADX trend strength filter to avoid whipsaws in ranging markets
# Discrete sizing (0.30) to balance profit potential and drawdown control
# Target: 100-200 total trades over 4 years (25-50/year) to stay within fee drag limits
# Donchian provides clear breakout levels; volume confirms institutional participation; ADX filters for trending regimes

name = "4h_Donchian20_Breakout_Volume_ADX_Regime"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate ADX on 1d data for regime filter
    # ADX calculation requires +DM, -DM, and TR
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range (TR)
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first value is NaN
    
    # Calculate +DM and -DM
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    def wilder_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    tr_smooth = wilder_smoothing(tr, period)
    plus_dm_smooth = wilder_smoothing(plus_dm, period)
    minus_dm_smooth = wilder_smoothing(minus_dm, period)
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smoothing(dx, period)
    
    # Align ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels on 4h data
    lookback = 20
    donchian_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(adx_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper band AND volume spike AND 1d ADX > 25 (trending)
            if (close[i] > donchian_upper[i] and 
                volume_filter[i] and 
                adx_1d_aligned[i] > 25):
                signals[i] = 0.30
                position = 1
            # Short conditions: price breaks below Donchian lower band AND volume spike AND 1d ADX > 25 (trending)
            elif (close[i] < donchian_lower[i] and 
                  volume_filter[i] and 
                  adx_1d_aligned[i] > 25):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian middle band OR touches Donchian lower band
            if close[i] < donchian_middle[i] or close[i] <= donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price crosses above Donchian middle band OR touches Donchian upper band
            if close[i] > donchian_middle[i] or close[i] >= donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals