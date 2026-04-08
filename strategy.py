#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels (1d)
    n1 = len(high_1d)
    camarilla_H4 = np.full(n1, np.nan)
    camarilla_L4 = np.full(n1, np.nan)
    camarilla_H3 = np.full(n1, np.nan)
    camarilla_L3 = np.full(n1, np.nan)
    
    for i in range(n1):
        if i < 1:
            continue
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        range_prev = high_prev - low_prev
        
        camarilla_H4[i] = close_prev + range_prev * 1.1 / 2
        camarilla_L4[i] = close_prev - range_prev * 1.1 / 2
        camarilla_H3[i] = close_prev + range_prev * 1.1 / 4
        camarilla_L3[i] = close_prev - range_prev * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    H4_12h = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    L4_12h = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    H3_12h = align_htf_to_ltf(prices, df_1d, camarilla_H3)
    L3_12h = align_htf_to_ltf(prices, df_1d, camarilla_L3)
    
    # 1d trend: 50-period EMA
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 12h volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h[i]) or np.isnan(H4_12h[i]) or np.isnan(L4_12h[i]) or 
            np.isnan(H3_12h[i]) or np.isnan(L3_12h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches H3 or trend fails
            if high[i] >= H3_12h[i] or close[i] < ema_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches L3 or trend fails
            if low[i] <= L3_12h[i] or close[i] > ema_50_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trend filter
            bullish = close[i] > ema_50_12h[i]
            bearish = close[i] < ema_50_12h[i]
            
            # Long: price touches L4 + bullish trend + volume
            if (low[i] <= L4_12h[i] and 
                bullish and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short: price touches H4 + bearish trend + volume
            elif (high[i] >= H4_12h[i] and 
                  bearish and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals