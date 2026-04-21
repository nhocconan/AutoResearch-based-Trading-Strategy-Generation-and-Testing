#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d ADX trend filter.
Long when price breaks above Donchian upper with volume > 1.5x average and daily ADX > 25.
Short when price breaks below Donchian lower with volume > 1.5x average and daily ADX > 25.
Exit when price returns to Donchian midpoint or volume drops below average.
This combines price channel breakout with volume and trend filters to reduce false signals
while maintaining low trade frequency (target: 20-30 trades/year) for robust performance
in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for ADX trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian upper and lower
    upper_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    mid_20 = (upper_20 + lower_20) / 2
    
    # Align Donchian levels to 4h
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, lower_20)
    mid_20_aligned = align_htf_to_ltf(prices, df_1d, mid_20)
    
    # Calculate daily ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
    
    # DI values
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    
    # Align daily ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(mid_20_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current price and volume
        price_close = prices['close'].iloc[i]
        vol_1d_current = align_htf_to_ltf(prices, df_1d, df_1d['volume'].values)[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian, volume surge, daily ADX > 25
            if (price_close > upper_20_aligned[i] and 
                vol_1d_current > 1.5 * vol_ma_20_aligned[i] and
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian, volume surge, daily ADX > 25
            elif (price_close < lower_20_aligned[i] and 
                  vol_1d_current > 1.5 * vol_ma_20_aligned[i] and
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to Donchian midpoint or volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: price <= midpoint or volume < average
                if (price_close <= mid_20_aligned[i] or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price >= midpoint or volume < average
                if (price_close >= mid_20_aligned[i] or
                    vol_1d_current < vol_ma_20_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Volume1.5x_DailyADX25"
timeframe = "4h"
leverage = 1.0