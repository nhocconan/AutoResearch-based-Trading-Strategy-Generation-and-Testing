#!/usr/bin/env python3
"""
6h_WickReversal_1wTrend_VolumeFilter
Hypothesis: On 6h timeframe, price rejection at weekly high/low zones (long upper/lower wicks) combined with weekly trend filter and volume confirmation captures reversal opportunities in both bull and bear markets. Weekly trend ensures alignment with higher timeframe momentum, reducing false signals in choppy conditions. Wick rejection indicates institutional defense of key levels, effective in ranging and trending markets.
"""
name = "6h_WickReversal_1wTrend_VolumeFilter"
timeframe = "6h"
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
    
    # Get weekly data for trend filter and rejection zones
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly trend: EMA21
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Weekly rejection zones: weekly high and low
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Align weekly data to 6h timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate wick ratios for current 6h bar
        body_size = abs(close[i] - prices['open'].iloc[i])
        total_range = high[i] - low[i]
        
        # Avoid division by zero
        if total_range == 0:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        upper_wick = high[i] - max(close[i], prices['open'].iloc[i])
        lower_wick = min(close[i], prices['open'].iloc[i]) - low[i]
        upper_wick_ratio = upper_wick / total_range
        lower_wick_ratio = lower_wick / total_range
        
        if position == 0:
            # Long: long lower wick (rejection of weekly low) + weekly uptrend + volume
            if (lower_wick_ratio > 0.4 and 
                low[i] <= weekly_low_aligned[i] * 1.001 and  # near weekly low
                close[i] > ema_21_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: long upper wick (rejection of weekly high) + weekly downtrend + volume
            elif (upper_wick_ratio > 0.4 and 
                  high[i] >= weekly_high_aligned[i] * 0.999 and  # near weekly high
                  close[i] < ema_21_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses weekly EMA21
            if position == 1:
                if close[i] < ema_21_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > ema_21_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals