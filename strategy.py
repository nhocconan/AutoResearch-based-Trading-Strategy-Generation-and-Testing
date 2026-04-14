#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR (14-period) for volatility filter
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6-hour Donchian channels (20-period)
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            donch_high[i] = np.max(high[i-19:i+1])
            donch_low[i] = np.min(low[i-19:i+1])
    
    # Calculate weekly high/low for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    weekly_high = np.full(len(df_1w), np.nan)
    weekly_low = np.full(len(df_1w), np.nan)
    
    if len(df_1w) >= 10:
        for i in range(9, len(df_1w)):
            weekly_high[i] = np.max(high_1w[i-9:i+1])
            weekly_low[i] = np.min(low_1w[i-9:i+1])
    
    weekly_high_6h = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_6h = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_6h[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(weekly_high_6h[i]) or
            np.isnan(weekly_low_6h[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_6h[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above weekly high = bullish bias, below weekly low = bearish bias
        price_above_weekly_high = close[i] > weekly_high_6h[i]
        price_below_weekly_low = close[i] < weekly_low_6h[i]
        
        if position == 0:
            # Long: Price breaks above 6h Donchian high AND above weekly high (bullish bias)
            if close[i] > donch_high[i] and price_above_weekly_high:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below 6h Donchian low AND below weekly low (bearish bias)
            elif close[i] < donch_low[i] and price_below_weekly_low:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 6h Donchian low OR below weekly low
            if close[i] < donch_low[i] or close[i] < weekly_low_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 6h Donchian high OR above weekly high
            if close[i] > donch_high[i] or close[i] > weekly_high_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Donchian_WeeklyTrend_Filter"
timeframe = "6h"
leverage = 1.0