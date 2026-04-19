#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with weekly pivot direction filter
# Uses weekly Camarilla pivot levels (from weekly high/low/close) to determine trend bias
# Long when price breaks above 6h Donchian upper band AND price > weekly R3 pivot
# Short when price breaks below 6h Donchian lower band AND price < weekly S3 pivot
# Volume confirmation filter to avoid false breakouts
# Target: 15-25 trades/year per symbol with high-probability entries
name = "6h_Donchian_WeeklyCamarilla_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly Camarilla pivots for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels: based on weekly high, low, close
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Camarilla formula: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    weekly_range = weekly_high - weekly_low
    r3_weekly = weekly_close + weekly_range * 1.1 / 4
    s3_weekly = weekly_close - weekly_range * 1.1 / 4
    
    # Align weekly pivots to 6h timeframe
    r3_weekly_aligned = align_htf_to_ltf(prices, df_1w, r3_weekly)
    s3_weekly_aligned = align_htf_to_ltf(prices, df_1w, s3_weekly)
    
    # 6h Donchian channels (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(r3_weekly_aligned[i]) or np.isnan(s3_weekly_aligned[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: break above Donchian high AND above weekly R3 pivot + volume
            if (close[i] > donchian_high[i] and 
                close[i] > r3_weekly_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND below weekly S3 pivot + volume
            elif (close[i] < donchian_low[i] and 
                  close[i] < s3_weekly_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: price breaks below Donchian low or weekly S3 pivot
            if close[i] < donchian_low[i] or close[i] < s3_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: price breaks above Donchian high or weekly R3 pivot
            if close[i] > donchian_high[i] or close[i] > r3_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals