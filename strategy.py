#!/usr/bin/env python3
name = "12h_1d_1w_VolumeWeighted_Camarilla_R3S3_Breakout"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla R3 and S3 levels
    R3 = np.full(len(high_1d), np.nan)
    S3 = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        if range_val > 0:  # Avoid division by zero
            R3[i] = prev_close + range_val * 1.1 / 4
            S3[i] = prev_close - range_val * 1.1 / 4
    
    # Get weekly trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20 = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_up = close_1w > ema20
    
    # Align HTF indicators to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, trend_up)
    
    # Volume-weighted price action confirmation (LTF)
    # Calculate 12-period VWAP-like momentum
    typical_price = (high + low + close) / 3
    vwap_num = np.zeros(n)
    vwap_den = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - 11)
        vwap_num[i] = np.sum(typical_price[start_idx:i+1] * volume[start_idx:i+1])
        vwap_den[i] = np.sum(volume[start_idx:i+1])
    
    vwap = np.divide(vwap_num, vwap_den, out=np.full_like(vwap_num, np.nan), where=vwap_den!=0)
    price_above_vwap = typical_price > vwap
    
    # Volume confirmation - current volume > 1.5x average of last 12 periods
    vol_ma12 = np.zeros(n)
    for i in range(n):
        if i < 12:
            vol_ma12[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma12[i] = np.mean(volume[i-11:i+1])
    
    volume_surge = volume > (1.5 * vol_ma12)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 12)  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or
            np.isnan(trend_up_aligned[i]) or
            np.isnan(vwap[i]) or
            np.isnan(vol_ma12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3 + uptrend + price above VWAP + volume surge
            if (close[i] > R3_aligned[i] and 
                trend_up_aligned[i] and 
                price_above_vwap[i] and 
                volume_surge[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + downtrend + price below VWAP + volume surge
            elif (close[i] < S3_aligned[i] and 
                  not trend_up_aligned[i] and 
                  not price_above_vwap[i] and 
                  volume_surge[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S3 or trend changes or price falls below VWAP
            if (close[i] < S3_aligned[i] or not trend_up_aligned[i] or not price_above_vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R3 or trend changes or price rises above VWAP
            if (close[i] > R3_aligned[i] or trend_up_aligned[i] or price_above_vwap[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals