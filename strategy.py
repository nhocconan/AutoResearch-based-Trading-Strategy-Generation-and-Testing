#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Strategy: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
# Long when price breaks above R1 and close > 1d EMA(34) and volume > 1.5x average
# Short when price breaks below S1 and close < 1d EMA(34) and volume > 1.5x average
# Exit when price returns to Pivot point
# Designed for 12h timeframe with selective entries to minimize trade frequency
# Works in both bull and bear markets via trend filter and volume confirmation

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Calculate 1d EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate average volume (20-period)
    vol_ma = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 20:
            vol_sum -= volume[i-20]
            vol_count -= 1
        if vol_count > 0:
            vol_ma[i] = vol_sum / vol_count
        else:
            vol_ma[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Ensure enough data for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or vol_ma[i] == 0.0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day
        if i >= 1:
            # Use previous day's high, low, close
            prev_idx = i - 1
            # For simplicity, use the same day's data approximation
            # In practice, we'd need to group by day, but this approximates
            prev_high = high[prev_idx]
            prev_low = low[prev_idx]
            prev_close = close[prev_idx]
            
            # Calculate pivot and Camarilla levels
            pivot = (prev_high + prev_low + prev_close) / 3.0
            range_val = prev_high - prev_low
            r1 = close + (range_val * 1.1 / 12)
            s1 = close - (range_val * 1.1 / 12)
        else:
            # Not enough data for previous day
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1, above 1d EMA34, volume spike
            if (close[i] > r1 and 
                close[i] > ema_34_aligned[i] and 
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1, below 1d EMA34, volume spike
            elif (close[i] < s1 and 
                  close[i] < ema_34_aligned[i] and 
                  volume[i] > 1.5 * vol_ma[i]):
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