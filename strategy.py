#!/usr/bin/env python3
name = "1d_Camarilla_Pivot_Squeeze_Trend"
timeframe = "1d"
leverage = 1.0

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
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly SMA200 for trend filter
    sma_200_1w = pd.Series(close_1w).rolling(window=200, min_periods=200).mean().values
    sma_200_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_200_1w)
    
    # Daily high, low, close for Camarilla calculation
    daily_high = prices['high'].values
    daily_low = prices['low'].values
    daily_close = prices['close'].values
    
    # Calculate Camarilla levels from previous day
    camarilla_high = np.roll(daily_high, 1)
    camarilla_low = np.roll(daily_low, 1)
    camarilla_close = np.roll(daily_close, 1)
    
    # Previous day's range
    prev_range = camarilla_high - camarilla_low
    
    # Camarilla levels (using previous day's data)
    r4 = camarilla_close + (prev_range * 1.1 / 2)
    r3 = camarilla_close + (prev_range * 1.1 / 4)
    r2 = camarilla_close + (prev_range * 1.1 / 6)
    r1 = camarilla_close + (prev_range * 1.1 / 12)
    s1 = camarilla_close - (prev_range * 1.1 / 12)
    s2 = camarilla_close - (prev_range * 1.1 / 6)
    s3 = camarilla_close - (prev_range * 1.1 / 4)
    s4 = camarilla_close - (prev_range * 1.1 / 2)
    
    # Bollinger Band width for squeeze detection (20-period)
    bb_middle = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = (bb_upper - bb_lower) / bb_middle
    
    # Squeeze condition: BB width below 20-period average
    bb_width_ma = pd.Series(bb_width).rolling(window=20, min_periods=20).mean().values
    squeeze = bb_width < bb_width_ma
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # ensure indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sma_200_1w_aligned[i]) or 
            np.isnan(squeeze[i]) or
            np.isnan(vol_filter[i]) or
            np.isnan(r1[i]) or np.isnan(s1[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 during squeeze + weekly uptrend + volume
            if close[i] > r1[i] and squeeze[i] and close[i] > sma_200_1w_aligned[i] and vol_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 during squeeze + weekly downtrend + volume
            elif close[i] < s1[i] and squeeze[i] and close[i] < sma_200_1w_aligned[i] and vol_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below S1 or squeeze breaks down
            if close[i] < s1[i] or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above R1 or squeeze breaks down
            if close[i] > r1[i] or not squeeze[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals