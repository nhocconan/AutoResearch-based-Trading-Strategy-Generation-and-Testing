#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 12h timeframe, enter long when price touches Camarilla S3 support with volume > 1.5x average and price > weekly EMA50, enter short when price touches Camarilla R3 resistance with volume > 1.5x average and price < weekly EMA50. Uses weekly trend filter to avoid counter-trend trades. Designed for 15-30 trades/year to minimize fee drag while capturing mean reversion in ranging markets and trend continuation in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "12h"
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
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_s3 = np.zeros(len(close_1d))
    camarilla_r3 = np.zeros(len(close_1d))
    pivot = np.zeros(len(close_1d))
    
    for i in range(len(close_1d)):
        # Camarilla formulas
        pivot[i] = (high_1d[i] + low_1d[i] + close_1d[i]) / 3
        range_ = high_1d[i] - low_1d[i]
        camarilla_s3[i] = close_1d[i] - (range_ * 1.1 / 4)
        camarilla_r3[i] = close_1d[i] + (range_ * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if data not available
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: price crosses below S3 or reverses from R3
            if close[i] < camarilla_s3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above R3 or reverses from S3
            if close[i] > camarilla_r3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price touches S3 support + price > weekly EMA50
                if (low[i] <= camarilla_s3_aligned[i] * 1.002 and  # Allow small buffer
                    close[i] > camarilla_s3_aligned[i] and
                    close[i] > ema_50_1w_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price touches R3 resistance + price < weekly EMA50
                elif (high[i] >= camarilla_r3_aligned[i] * 0.998 and  # Allow small buffer
                      close[i] < camarilla_r3_aligned[i] and
                      close[i] < ema_50_1w_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals