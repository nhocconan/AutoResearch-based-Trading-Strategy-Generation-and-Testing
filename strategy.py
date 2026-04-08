# 12h_1d1w_camarilla_volume_trend_v1
# Hypothesis: 12h Camarilla pivot strategy with 1d/1h trend filters and volume confirmation. 
# Long when price crosses above H3 with volume > 2x average and price > 1h EMA20
# Short when price crosses below L3 with volume > 2x average and price < 1h EMA20
# Exit on opposite level cross or volume drop below average
# Uses tight 12h timeframe to target 15-30 trades/year, avoiding fee drag
# Works in bull (trend following) and bear (mean reversion at extremes) via dual filters

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d1w_camarilla_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    pivot = (high + low + close) / 3.0
    range_val = high - low
    H3 = pivot + (range_val * 1.1 / 4)
    L3 = pivot - (range_val * 1.1 / 4)
    return H3, L3

def calculate_ema(arr, period):
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    ema = np.full_like(arr, np.nan, dtype=float)
    alpha = 2.0 / (period + 1)
    ema[period-1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
    return ema

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Get 1h data for trend filter
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Camarilla levels
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    camarilla_H3_12h, camarilla_L3_12h = calculate_camarilla(high_12h, low_12h, close_12h)
    
    # Calculate 1h EMA20 for trend filter
    close_1h = df_1h['close'].values
    ema_20_1h = calculate_ema(close_1h, 20)
    
    # Align indicators to 12h timeframe
    camarilla_H3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_H3_12h)
    camarilla_L3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_L3_12h)
    ema_20_1h_aligned = align_htf_to_ltf(prices, df_1h, ema_20_1h)
    
    # Volume confirmation: 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(camarilla_H3_12h_aligned[i]) or np.isnan(camarilla_L3_12h_aligned[i]) or
            np.isnan(ema_20_1h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        H3 = camarilla_H3_12h_aligned[i]
        L3 = camarilla_L3_12h_aligned[i]
        trend_up = price > ema_20_1h_aligned[i]
        
        if position == 1:  # Long
            if price < L3 or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            if price > H3 or vol_ratio < 1.0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if price > H3 and vol_ratio > 2.0 and trend_up:
                position = 1
                signals[i] = 0.25
            elif price < L3 and vol_ratio > 2.0 and not trend_up:
                position = -1
                signals[i] = -0.25
    
    return signals