#!/usr/bin/env python3
# 12h_Camarilla_R1S1_Breakout_1dTrend_Volume
# Strategy: Trade Camarilla R1/S1 breakouts on 12h with 1d EMA trend filter and volume confirmation
# Long when price breaks above R1 with volume > 1.5x average and price > 1d EMA50
# Short when price breaks below S1 with volume > 1.5x average and price < 1d EMA50
# Exit when price returns to Camarilla pivot level
# Designed for 12h timeframe with selective entries to minimize trade frequency and maximize edge

name = "12h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate average volume for confirmation (20-period)
    vol_avg = np.zeros(n)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels for current 12h bar
        # Using previous bar's high, low, close
        if i == 0:
            continue
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        # Camarilla levels
        range_val = ph - pl
        r1 = pc + (range_val * 1.1 / 12)
        s1 = pc - (range_val * 1.1 / 12)
        pivot = pc
        
        if position == 0:
            # Enter long: price breaks above R1 with volume confirmation and uptrend filter
            if close[i] > r1 and volume[i] > 1.5 * vol_avg[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 with volume confirmation and downtrend filter
            elif close[i] < s1 and volume[i] > 1.5 * vol_avg[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot or below
            if close[i] <= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot or above
            if close[i] >= pivot:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals