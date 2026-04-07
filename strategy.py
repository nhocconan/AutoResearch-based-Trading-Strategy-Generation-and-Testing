#!/usr/bin/env python3
"""
4h_camarilla_pivot_12h_volume_v1
Hypothesis: On 4h timeframe, use daily Camarilla pivot levels for mean reversion entries when price touches S3/R3 levels, filtered by 12h EMA trend direction and volume confirmation. This strategy captures reversals at key support/resistance levels while avoiding counter-trend trades. Camarilla levels work well in ranging markets, and the 12h EMA filter ensures alignment with higher timeframe trend. Volume confirmation reduces false signals. Target: 80-150 trades over 4 years (20-38/year) to balance opportunity with cost efficiency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_12h_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate daily Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_S3 = np.zeros(len(df_1d))
    camarilla_S4 = np.zeros(len(df_1d))
    camarilla_R3 = np.zeros(len(df_1d))
    camarilla_R4 = np.zeros(len(df_1d))
    
    for i in range(len(df_1d)):
        if i == 0:
            # Use first available values
            high_val = high_1d[i]
            low_val = low_1d[i]
            close_val = close_1d[i]
        else:
            high_val = high_1d[i]
            low_val = low_1d[i]
            close_val = close_1d[i]
        
        # Camarilla formulas
        range_val = high_val - low_val
        camarilla_S3[i] = close_val - (range_val * 1.1 / 6)
        camarilla_S4[i] = close_val - (range_val * 1.1 / 4)
        camarilla_R3[i] = close_val + (range_val * 1.1 / 6)
        camarilla_R4[i] = close_val + (range_val * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    
    # Calculate volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(camarilla_R3_aligned[i]) or np.isnan(volume_ma[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 12h EMA direction
        if i >= 21:
            ema_prev = ema_12h_aligned[i-1]
            ema_curr = ema_12h_aligned[i]
            uptrend = ema_curr > ema_prev
        else:
            uptrend = True  # Default to allow trading if insufficient history
        
        # Volume confirmation: current volume > 1.5x average
        volume_ok = volume[i] > (volume_ma[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: price reaches S4 (strong support) or closes below S3
            if close[i] <= camarilla_S4_aligned[i] or close[i] < camarilla_S3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches R4 (strong resistance) or closes above R3
            if close[i] >= camarilla_R4_aligned[i] or close[i] > camarilla_R3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches S3 level with volume confirmation in uptrend
            if (uptrend and volume_ok and 
                abs(high[i] - camarilla_S3_aligned[i]) < (high[i] * 0.001)):  # Within 0.1% of S3
                position = 1
                signals[i] = 0.25
            # Short entry: price touches R3 level with volume confirmation in downtrend
            elif ((not uptrend) and volume_ok and 
                  abs(low[i] - camarilla_R3_aligned[i]) < (low[i] * 0.001)):  # Within 0.1% of R3
                position = -1
                signals[i] = -0.25
    
    return signals