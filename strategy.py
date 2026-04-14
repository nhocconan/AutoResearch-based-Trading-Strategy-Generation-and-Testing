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
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR for volatility filter (14-period)
    tr = np.zeros(len(df_1w))
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    atr_6h = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate weekly high/low for breakout levels
    weekly_high = np.full(len(df_1w), np.nan)
    weekly_low = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        weekly_high[i] = high_1w[i]
        weekly_low[i] = low_1w[i]
    
    weekly_high_6h = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_6h = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Calculate 6-hour moving average for trend filter (20-period)
    ma_6h = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            ma_6h[i] = np.mean(close[i-19:i+1])
    
    # Calculate volume moving average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_6h[i]) or
            np.isnan(weekly_high_6h[i]) or
            np.isnan(weekly_low_6h[i]) or
            np.isnan(ma_6h[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility periods (ATR < 0.5% of price)
        if atr_6h[i] < 0.005 * close[i]:
            signals[i] = 0.0
            continue
        
        # Volume ratio: current volume vs 20-period average
        if vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        # Volume threshold: require significant spike
        vol_threshold = 2.5
        
        if position == 0:
            # Long: Price breaks above weekly high with volume confirmation and above 6h MA
            if close[i] > weekly_high_6h[i] and volume_ratio > vol_threshold and close[i] > ma_6h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below weekly low with volume confirmation and below 6h MA
            elif close[i] < weekly_low_6h[i] and volume_ratio > vol_threshold and close[i] < ma_6h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below 6h MA OR below weekly low
            if close[i] < ma_6h[i] or close[i] < weekly_low_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above 6h MA OR above weekly high
            if close[i] > ma_6h[i] or close[i] > weekly_high_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_WeeklyBreakout_MA_Volume"
timeframe = "6h"
leverage = 1.0