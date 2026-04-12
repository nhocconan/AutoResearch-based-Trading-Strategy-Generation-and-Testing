#!/usr/bin/env python3
"""
12h_1d_Bollinger_Breakout_TrendFilter
Hypothesis: Uses daily Bollinger Bands with breakout confirmation on 12h timeframe, filtered by 1d ADX trend strength and volume.
Designed to capture strong trend continuation moves while avoiding whipsaws in ranging markets. Works in both bull and bear by requiring strong trend (ADX>25) and volume expansion.
Target: 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Bollinger_Breakout_TrendFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === DAILY BOLLINGER BANDS ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 20-period SMA and standard deviation
    sma_20 = np.full(len(close_1d), np.nan)
    std_20 = np.full(len(close_1d), np.nan)
    
    for i in range(19, len(close_1d)):
        sma_20[i] = np.mean(close_1d[i-19:i+1])
        std_20[i] = np.std(close_1d[i-19:i+1])
    
    # Upper and lower bands (2 standard deviations)
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    # Align Bollinger Bands to 12h timeframe
    upper_12h = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_12h = align_htf_to_ltf(prices, df_1d, lower_band)
    sma_12h = align_htf_to_ltf(prices, df_1d, sma_20)
    
    # === 1d ADX TREND FILTER ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and Directional Movement
    tr_1d = np.zeros(len(close_1d))
    plus_dm_1d = np.zeros(len(close_1d))
    minus_dm_1d = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
        
        plus_dm_1d[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm_1d[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
    
    # Wilder's smoothing
    def WilderSmoothing(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[1:period+1])
        for i in range(period, len(arr)):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    period = 14
    tr_smooth_1d = WilderSmoothing(tr_1d, period)
    plus_di_1d = 100 * WilderSmoothing(plus_dm_1d, period) / tr_smooth_1d
    minus_di_1d = 100 * WilderSmoothing(minus_dm_1d, period) / tr_smooth_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = WilderSmoothing(dx_1d, period)
    
    # Align ADX to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # === VOLUME CONFIRMATION (12h) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]) or 
            np.isnan(adx_12h[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(sma_12h[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions
        # Long: Price breaks above upper Bollinger Band with volume + strong trend
        long_breakout = (close[i] > upper_12h[i]) and (vol_ratio[i] > 1.8) and (adx_12h[i] > 25)
        
        # Short: Price breaks below lower Bollinger Band with volume + strong trend
        short_breakout = (close[i] < lower_12h[i]) and (vol_ratio[i] > 1.8) and (adx_12h[i] > 25)
        
        # Exit: Price returns to middle Bollinger Band or trend weakens
        exit_long = (position == 1) and ((close[i] < sma_12h[i]) or (adx_12h[i] < 20))
        exit_short = (position == -1) and ((close[i] > sma_12h[i]) or (adx_12h[i] < 20))
        
        # Execute trades
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals