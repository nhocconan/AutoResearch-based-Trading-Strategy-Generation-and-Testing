#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla R3/S3 mean reversion with 12h volume regime filter.
Long when price touches S3 level AND 12h volume ratio > 1.3 AND close > open (bullish rejection candle).
Short when price touches R3 level AND 12h volume ratio > 1.3 AND close < open (bearish rejection candle).
Exit at R1/S1 levels or opposite R3/S3 touch.
Uses Camarilla pivot levels from 1d for structure and 12h HTF for volume regime to avoid low-volume false signals.
Target: 50-150 total trades over 4 years (12-37/year). Camarilla R3/S3 often act as support/resistance with reversal potential.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels using previous day's OHLC
    pivot = (high_1d[:-1] + low_1d[:-1] + close_1d[:-1]) / 3.0
    range_1d = high_1d[:-1] - low_1d[:-1]
    
    # Camarilla levels
    r3 = pivot + range_1d * 1.1 / 4
    s3 = pivot - range_1d * 1.1 / 4
    r1 = pivot + range_1d * 1.1 / 12
    s1 = pivot - range_1d * 1.1 / 12
    
    # Align 1d Camarilla levels to 6h timeframe
    r3_6h = align_htf_to_ltf(prices, df_1d[:-1], r3)  # use previous day's levels
    s3_6h = align_htf_to_ltf(prices, df_1d[:-1], s3)
    r1_6h = align_htf_to_ltf(prices, df_1d[:-1], r1)
    s1_6h = align_htf_to_ltf(prices, df_1d[:-1], s1)
    
    # Calculate 12h volume ratio for regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = align_htf_to_ltf(prices, df_12h, vol_12h / vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # volume MA needs 20 periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or 
            np.isnan(vol_ratio_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ratio = vol_ratio_12h[i]
        is_bullish = close[i] > open_price if (open_price := prices['open'].iloc[i]) else False
        is_bearish = close[i] < open_price if (open_price := prices['open'].iloc[i]) else False
        
        if position == 0:
            # Long: Price touches S3 AND high volume regime AND bullish rejection
            if price <= s3_6h[i] and vol_ratio > 1.3 and is_bullish:
                signals[i] = 0.25
                position = 1
            # Short: Price touches R3 AND high volume regime AND bearish rejection
            elif price >= r3_6h[i] and vol_ratio > 1.3 and is_bearish:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            if position == 1:  # Long position
                if price >= r1_6h[i] or price <= s3_6h[i]:  # Profit at R1 or stop at S3
                    exit_signal = True
            else:  # Short position
                if price <= s1_6h[i] or price >= r3_6h[i]:  # Profit at S1 or stop at R3
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Camarilla_R3S3_MeanReversion_12hVolumeRegime_RejectionCandle"
timeframe = "6h"
leverage = 1.0