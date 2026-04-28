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
    
    # Get weekly data for trend (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR(10) for trend filter
    tr1 = np.maximum(high_1w[1:], low_1w[:-1]) - np.minimum(high_1w[1:], low_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.inf], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate weekly close SMA(20) for trend direction
    sma20_1w = pd.Series(close_1w).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly ATR and SMA to daily
    atr_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    sma20_aligned = align_htf_to_ltf(prices, df_1w, sma20_1w)
    
    # Calculate daily Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume SMA(20) for volume filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(sma20_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly SMA20 with ATR buffer
        uptrend = close[i] > sma20_aligned[i] + 0.5 * atr_aligned[i]
        downtrend = close[i] < sma20_aligned[i] - 0.5 * atr_aligned[i]
        
        # Volume filter: volume above average
        vol_filter = volume[i] > vol_sma[i]
        
        # Breakout conditions
        long_breakout = high[i] > highest_high[i-1]  # break above previous 20-day high
        short_breakout = low[i] < lowest_low[i-1]    # break below previous 20-day low
        
        # Long conditions: uptrend + volume + long breakout
        long_condition = uptrend and vol_filter and long_breakout
        
        # Short conditions: downtrend + volume + short breakout
        short_condition = downtrend and vol_filter and short_breakout
        
        if long_condition and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_condition and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit conditions: opposite breakout or trend reversal
        elif position == 1 and (downtrend or short_breakout):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (uptrend or long_breakout):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WeeklyTrend_Donchian20_Volume"
timeframe = "1d"
leverage = 1.0