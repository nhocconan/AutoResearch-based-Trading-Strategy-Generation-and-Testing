#!/usr/bin/env python3
# 1h_Aroon_Trend_4hATR_Volume_Confirmation
# Hypothesis: Aroon(25) identifies strong trends on 1h with 4h ATR-based position sizing and volume confirmation.
# Aroon measures trend strength and direction, filtering for trending markets while avoiding ranging conditions.
# 4h ATR provides volatility-based sizing to adapt to market conditions, reducing risk in high volatility.
# Volume confirmation ensures breakouts have institutional participation.
# Designed for 1h timeframe with 15-35 trades/year using tight entry conditions.

name = "1h_Aroon_Trend_4hATR_Volume_Confirmation"
timeframe = "1h"
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
    
    # 4h data for ATR-based volatility sizing
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Aroon indicator (25-period) for trend strength and direction
    def calculate_aroon(high_arr, low_arr, period):
        n = len(high_arr)
        aroon_up = np.full(n, np.nan)
        aroon_down = np.full(n, np.nan)
        
        for i in range(period - 1, n):
            # Find highest high and lowest low in the lookback period
            highest_high_idx = np.argmax(high_arr[i - period + 1:i + 1]) + (i - period + 1)
            lowest_low_idx = np.argmin(low_arr[i - period + 1:i + 1]) + (i - period + 1)
            
            aroon_up[i] = ((period - 1) - (i - highest_high_idx)) / (period - 1) * 100
            aroon_down[i] = ((period - 1) - (i - lowest_low_idx)) / (period - 1) * 100
            
        return aroon_up, aroon_down
    
    aroon_up, aroon_down = calculate_aroon(high, low, 25)
    
    # 4h ATR for volatility-based position sizing
    def calculate_atr(high_arr, low_arr, close_arr, period):
        n = len(high_arr)
        tr = np.full(n, np.nan)
        atr = np.full(n, np.nan)
        
        for i in range(n):
            if i == 0:
                tr[i] = high_arr[i] - low_arr[i]
            else:
                tr[i] = max(high_arr[i] - low_arr[i], 
                           abs(high_arr[i] - close_arr[i-1]),
                           abs(low_arr[i] - close_arr[i-1]))
        
        # Calculate ATR using Wilder's smoothing
        if n >= period:
            atr[period-1] = np.mean(tr[0:period])
            for i in range(period, n):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
                
        return atr
    
    atr_4h = calculate_atr(high_4h, low_4h, close_4h, 14)
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, period):
        res = np.full_like(arr, np.nan)
        if len(arr) >= period:
            for i in range(period - 1, len(arr)):
                res[i] = np.mean(arr[i - period + 1:i + 1])
        return res
    
    vol_ma_20 = mean_arr(volume, 20)
    
    # Align 4h ATR to 1h timeframe (wait for 4h bar to close)
    atr_4h_aligned = align_htf_to_ltf(prices, df_4h, atr_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for Aroon and ATR
    
    for i in range(start_idx, n):
        if np.isnan(aroon_up[i]) or np.isnan(aroon_down[i]) or \
           np.isnan(atr_4h_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate position size based on 4h ATR (inverse volatility)
        # Normalize ATR to get volatility factor (lower ATR = higher size)
        atr_normalized = atr_4h_aligned[i] / np.nanmean(atr_4h_aligned[max(0, i-50):i+1])
        vol_factor = np.clip(1.0 / (atr_normalized + 0.001), 0.5, 2.0)  # Limit volatility adjustment
        base_size = 0.20
        position_size = base_size * vol_factor
        
        if position == 0:
            # Long: Aroon up > 70 and Aroon down < 30 (strong uptrend) + volume confirmation
            if aroon_up[i] > 70 and aroon_down[i] < 30 and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = position_size
                position = 1
            # Short: Aroon down > 70 and Aroon up < 30 (strong downtrend) + volume confirmation
            elif aroon_down[i] > 70 and aroon_up[i] < 30 and volume[i] > 1.5 * vol_ma_20[i]:
                signals[i] = -position_size
                position = -1
        elif position == 1:
            # Long exit: Trend weakening (Aroon down > 50) or volume drops significantly
            if aroon_down[i] > 50 or volume[i] < 0.5 * vol_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = position_size
        elif position == -1:
            # Short exit: Trend weakening (Aroon up > 50) or volume drops significantly
            if aroon_up[i] > 50 or volume[i] < 0.5 * vol_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -position_size
    
    return signals