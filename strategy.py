#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 12h volatility filter and 1d trend confirmation
# Long when price breaks above upper Donchian band, 12h ATR < median (low volatility), and 1d EMA50 uptrend
# Short when price breaks below lower Donchian band, 12h ATR < median (low volatility), and 1d EMA50 downtrend
# Exit when price returns to the middle of the Donchian channel or reverses to opposite band
# Designed to capture breakouts during low volatility periods with trend confirmation to avoid whipsaws
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "6h_Donchian20_12hATRFilter_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_donchian = high_series.rolling(window=20, min_periods=20).max().values
    lower_donchian = low_series.rolling(window=20, min_periods=20).min().values
    middle_donchian = (upper_donchian + lower_donchian) / 2
    
    # Calculate 12h ATR for volatility filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 1:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = high_12h[0] - low_12h[0]  # First period
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_median_12h = pd.Series(atr_12h).rolling(window=50, min_periods=50).median().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    atr_median_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_median_12h)
    low_volatility = atr_12h_aligned < atr_median_12h_aligned
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for ATR calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(low_volatility[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper Donchian, low volatility, EMA50 uptrend
            if (close[i] > upper_donchian[i] and 
                low_volatility[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower Donchian, low volatility, EMA50 downtrend
            elif (close[i] < lower_donchian[i] and 
                  low_volatility[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to middle or breaks below lower band
            if (close[i] <= middle_donchian[i]) or (close[i] < lower_donchian[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to middle or breaks above upper band
            if (close[i] >= middle_donchian[i]) or (close[i] > upper_donchian[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals