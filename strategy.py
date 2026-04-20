#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 12h data for primary trend and volatility
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h Donchian channels (20-period) for breakout signals
    donchian_window = 20
    upper_12h = pd.Series(high_12h).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_12h = pd.Series(low_12h).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # 12h ATR for volatility filtering
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h Volume confirmation (current vs 20-period average)
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / np.where(vol_ma_20_12h == 0, 1, vol_ma_20_12h)
    
    # Align 12h indicators to 6h timeframe
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, upper_12h)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, lower_12h)
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # Load 1d data for weekly pivot context
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d weekly pivot points (using prior week's data)
    # Calculate weekly high/low/close from daily data
    # We'll approximate weekly by using 5-day aggregates
    weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
    weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
    weekly_close = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
    
    # Weekly pivot: (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R1: 2*P - L
    weekly_r1 = 2 * weekly_pivot - weekly_low
    # Weekly S1: 2*P - H
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivots to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN in critical values
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(atr_12h_aligned[i]) or np.isnan(vol_ratio_12h_aligned[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = upper_12h_aligned[i]
        lower = lower_12h_aligned[i]
        atr = atr_12h_aligned[i]
        vol_ratio = vol_ratio_12h_aligned[i]
        pivot = weekly_pivot_aligned[i]
        r1 = weekly_r1_aligned[i]
        s1 = weekly_s1_aligned[i]
        
        # Volatility filter: avoid extreme volatility
        atr_ma_50 = pd.Series(atr_12h_aligned).rolling(window=50, min_periods=50).mean().values[i]
        vol_filter = atr < 2.0 * atr_ma_50
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio > 1.2)
        
        if position == 0:
            # Enter long: price breaks above 12h Donchian upper AND above weekly R1
            if price > upper and price > r1 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below 12h Donchian lower AND below weekly S1
            elif price < lower and price < s1 and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 12h Donchian lower OR below weekly pivot
            if price < lower or price < pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 12h Donchian upper OR above weekly pivot
            if price > upper or price > pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_1d_DonchianWeeklyPivot_Breakout"
timeframe = "6h"
leverage = 1.0