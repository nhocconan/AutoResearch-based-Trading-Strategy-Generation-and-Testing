#!/usr/bin/env python3
# 12h_1d_ema_touch_volume_v1
# Hypothesis: 12-hour price touching 21-period EMA with volume > 1.8x average and alignment with daily trend (price > daily EMA200).
# Long when price touches EMA21 from below (low <= EMA21 < close) with volume confirmation and bullish daily trend.
# Short when price touches EMA21 from above (high >= EMA21 > close) with volume confirmation and bearish daily trend.
# Exit when price crosses EMA21 in opposite direction.
# Uses EMA21 for dynamic support/resistance and daily EMA200 for trend filter to work in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_ema_touch_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for daily EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate EMA21 for 12h timeframe
    close_s = pd.Series(close)
    ema21 = close_s.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate daily EMA200
    close_1d_s = pd.Series(df_1d['close'].values)
    ema200_1d = close_1d_s.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMA200 to 12h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(21, n):  # Start after EMA21 warmup
        # Skip if data not ready
        if (np.isnan(ema21[i]) or np.isnan(ema200_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        ema21_val = ema21[i]
        daily_ema200 = ema200_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: price crosses below EMA21
            if price < ema21_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price crosses above EMA21
            if price > ema21_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price touches EMA21 from below with volume expansion and bullish daily trend
            if low[i] <= ema21_val < close[i] and vol_ratio > 1.8 and price > daily_ema200:
                position = 1
                signals[i] = 0.25
            # Enter short: price touches EMA21 from above with volume expansion and bearish daily trend
            elif high[i] >= ema21_val > close[i] and vol_ratio > 1.8 and price < daily_ema200:
                position = -1
                signals[i] = -0.25
    
    return signals