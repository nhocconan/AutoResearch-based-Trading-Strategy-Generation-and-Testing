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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate SMA(50) on 1w close
    close_1w = df_1w['close'].values
    sma_period = 50
    sma_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= sma_period:
        for i in range(sma_period-1, len(close_1w)):
            sma_1w[i] = np.mean(close_1w[i-sma_period+1:i+1])
    
    # Align 1w SMA to 12h timeframe
    sma_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_1w)
    
    # Get 1d data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 1d high/low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_period = 20
    upper_1d = np.full(len(high_1d), np.nan)
    lower_1d = np.full(len(low_1d), np.nan)
    if len(high_1d) >= donch_period:
        for i in range(donch_period-1, len(high_1d)):
            upper_1d[i] = np.max(high_1d[i-donch_period+1:i+1])
            lower_1d[i] = np.min(low_1d[i-donch_period+1:i+1])
    
    # Align 1d Donchian to 12h timeframe
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need SMA (50), Donchian (20), volume MA (20)
    start_idx = max(sma_period, donch_period, vol_ma_period)
    
    for i in range(start_idx, n):
        if (np.isnan(sma_1w_aligned[i]) or
            np.isnan(upper_1d_aligned[i]) or
            np.isnan(lower_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Trend filter: price above/below 1w SMA(50)
        uptrend = price > sma_1w_aligned[i]
        downtrend = price < sma_1w_aligned[i]
        
        # Volume confirmation: > 1.5x average volume
        volume_confirmation = vol_ratio > 1.5
        
        if position == 0:
            # Long entry: price breaks above 1d Donchian upper in uptrend with volume
            if price > upper_1d_aligned[i] and uptrend and volume_confirmation:
                signals[i] = size
                position = 1
            # Short entry: price breaks below 1d Donchian lower in downtrend with volume
            elif price < lower_1d_aligned[i] and downtrend and volume_confirmation:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below 1d Donchian lower or trend reverses
            if price < lower_1d_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above 1d Donchian upper or trend reverses
            if price > upper_1d_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_1wSMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0