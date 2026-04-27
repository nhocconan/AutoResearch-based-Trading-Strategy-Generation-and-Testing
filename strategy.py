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
    
    # Get 1d and 1w data for HTF context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week OHLC)
    high_1w = get_htf_data(prices, '1w')['high'].values
    low_1w = get_htf_data(prices, '1w')['low'].values
    close_1w = get_htf_data(prices, '1w')['close'].values
    
    high_1w_prev = np.roll(high_1w, 1)
    low_1w_prev = np.roll(low_1w, 1)
    close_1w_prev = np.roll(close_1w, 1)
    high_1w_prev[0] = np.nan
    low_1w_prev[0] = np.nan
    close_1w_prev[0] = np.nan
    
    weekly_pivot = (high_1w_prev + low_1w_prev + close_1w_prev) / 3.0
    weekly_r1 = 2 * weekly_pivot - low_1w_prev
    weekly_s1 = 2 * weekly_pivot - high_1w_prev
    
    # Align weekly pivots to 6h
    weekly_pivot_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1w'), weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1w'), weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '1w'), weekly_s1)
    
    # 60-period moving average on 6h for trend filter
    ma_60 = np.full(n, np.nan)
    for i in range(59, n):
        ma_60[i] = np.mean(close[i-59:i+1])
    
    # Volume filter: volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly pivot (1), MA60 (60), volume MA (20)
    start_idx = max(60, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(ma_60[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Weekly trend filter
        bullish_weekly = price > weekly_pivot_aligned[i]
        bearish_weekly = price < weekly_pivot_aligned[i]
        
        # Price relative to 60-period MA
        above_ma = price > ma_60[i]
        below_ma = price < ma_60[i]
        
        if position == 0:
            # Long: price crosses above weekly S1 with volume and above weekly pivot and above MA60
            if price > weekly_s1_aligned[i] and vol_filter and bullish_weekly and above_ma:
                signals[i] = size
                position = 1
            # Short: price crosses below weekly R1 with volume and below weekly pivot and below MA60
            elif price < weekly_r1_aligned[i] and vol_filter and bearish_weekly and below_ma:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly pivot or below MA60
            if price < weekly_pivot_aligned[i] or price < ma_60[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above weekly pivot or above MA60
            if price > weekly_pivot_aligned[i] or price > ma_60[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_WeeklyPivot_S1R1_MA60_Volume"
timeframe = "6h"
leverage = 1.0