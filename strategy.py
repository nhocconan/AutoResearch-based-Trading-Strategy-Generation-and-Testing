#!/usr/bin/env python3
name = "12h_Wyckoff_Spring_1WeekTrend_Volume"
timeframe = "12h"
leverage = 1.0

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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 20-period EMA for weekly trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly EMA20 to 12h timeframe
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate 12-period high and low for Wyckoff Spring pattern
    period_high = np.zeros(n)
    period_low = np.zeros(n)
    
    for i in range(12, n):
        period_high[i] = np.max(high[i-12:i])
        period_low[i] = np.min(low[i-12:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(12, n):
        # Skip if weekly EMA data not ready
        if np.isnan(ema20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        uptrend = close[i] > ema20_1w_aligned[i]
        downtrend = close[i] < ema20_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 12-period average volume
        avg_volume = np.mean(volume[i-12:i])
        volume_confirm = volume[i] > avg_volume * 1.3
        
        if position == 0:
            # Wyckoff Spring: price tests below recent low but closes back above it with volume
            spring_long = (low[i] < period_low[i]) and (close[i] > period_low[i]) and volume_confirm
            # Wyckoff Upthrust: price tests above recent high but closes back below it with volume
            upthrust_short = (high[i] > period_high[i]) and (close[i] < period_high[i]) and volume_confirm
            
            if spring_long and uptrend:
                signals[i] = 0.25
                position = 1
            elif upthrust_short and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 6-period low
            exit_low = np.min(low[i-6:i])
            if close[i] < exit_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 6-period high
            exit_high = np.max(high[i-6:i])
            if close[i] > exit_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals