#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 1-day Donchian breakout with weekly EMA200 trend filter and volume confirmation
    # Captures breakouts from 20-day price channels while filtering by long-term trend
    # Volume surge confirms breakout strength, weekly EMA200 filters trend direction
    # Works in both bull and bear markets by only taking trend-aligned breakouts
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly EMA200 trend filter
    ema_1w_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1w_200_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_200)
    
    # Daily Donchian Channel (20-period)
    high_20 = pd.Series(prices['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(prices['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (20-period MA surge)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_200_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-day high with volume surge AND weekly EMA200 uptrend
            if prices['high'].values[i] > high_20[i] and vol_surge[i] and prices['close'].values[i] > ema_1w_200_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low with volume surge AND weekly EMA200 downtrend
            elif prices['low'].values[i] < low_20[i] and vol_surge[i] and prices['close'].values[i] < ema_1w_200_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to 20-day midpoint or opposite breakout
            midpoint = (high_20[i] + low_20[i]) / 2
            if position == 1:
                if prices['close'].values[i] < midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if prices['close'].values[i] > midpoint:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_WeeklyEMA200_Trend_VolumeSurge_v1"
timeframe = "1d"
leverage = 1.0