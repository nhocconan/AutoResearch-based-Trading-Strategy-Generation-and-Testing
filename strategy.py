#!/usr/bin/env python3
name = "1d_Weekly_Camarilla_R3_S3_Breakout_Trend"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1w data for Camarilla R3 and S3 levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla R3 = Close + (High - Low) * 1.1 / 4
    # Camarilla S3 = Close - (High - Low) * 1.1 / 4
    R3_1w = close_1w + (high_1w - low_1w) * 1.1 / 4
    S3_1w = close_1w - (high_1w - low_1w) * 1.1 / 4
    
    # Align to 1d
    R3_1w_aligned = align_htf_to_ltf(prices, df_1w, R3_1w)
    S3_1w_aligned = align_htf_to_ltf(prices, df_1w, S3_1w)
    
    # Trend filter: price above/below 20-period SMA
    close_series = pd.Series(close)
    sma_20 = close_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if (np.isnan(R3_1w_aligned[i]) or np.isnan(S3_1w_aligned[i]) or 
            np.isnan(sma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3, uptrend (price > SMA20)
            if (close[i] > R3_1w_aligned[i] and 
                close[i] > sma_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3, downtrend (price < SMA20)
            elif (close[i] < S3_1w_aligned[i] and 
                  close[i] < sma_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below S3 (opposite level)
            if close[i] < S3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above R3 (opposite level)
            if close[i] > R3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals