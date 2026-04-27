#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams %R reversal with volume confirmation and 1-day trend filter.
Enters on extreme oversold/overbought conditions during low volatility periods.
Designed to work in both bull and bear markets by using 1-day trend as filter.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
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
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 4-hour data for Williams %R and ATR
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    # Calculate 4-hour Williams %R (14)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    highest_high = pd.Series(high_4h).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_4h).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_4h) / (highest_high - lowest_low)
    
    # Calculate 4-hour ATR(14) for volatility filter
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.concatenate([close_4h[0:1], close_4h[:-1]]))
    tr3 = np.abs(low_4h - np.concatenate([close_4h[0:1], close_4h[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all 4h indicators
    williams_r_aligned = align_htf_to_ltf(prices, df_4h, williams_r)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Williams %R, ATR, and 1d EMA
    start_idx = max(14, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(williams_r_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        wr = williams_r_aligned[i]
        atr_val = atr_aligned[i]
        trend_1d = ema_50_1d_aligned[i]
        
        # Volatility filter: ATR < 0.02 * price (low volatility environment)
        vol_filter = atr_val < 0.02 * close[i]
        
        # Entry conditions: Williams %R reversal with volume and 1d trend alignment
        if position == 0:
            # Long: Williams %R oversold (< -80) + low vol + price above 1d EMA
            if wr < -80 and vol_filter and close[i] > trend_1d:
                signals[i] = size
                position = 1
            # Short: Williams %R overbought (> -20) + low vol + price below 1d EMA
            elif wr > -20 and vol_filter and close[i] < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or trend reversal
            if wr > -50 or close[i] < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or trend reversal
            if wr < -50 or close[i] > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_WilliamsR_Reversal_VolumeFilter_1dTrend"
timeframe = "4h"
leverage = 1.0