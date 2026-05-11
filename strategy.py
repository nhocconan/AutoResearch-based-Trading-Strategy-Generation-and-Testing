#!/usr/bin/env python3
name = "4h_HTF_1d_Camarilla_R1S1_Breakout_Trend_Filtered"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1_1d = close_1d + (range_1d * 1.0833)
    s1_1d = close_1d - (range_1d * 1.0833)
    
    # Align levels to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (4h volume > 1.8x 30-period average)
    volume_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > 1.8 * volume_ma30
    
    # Choppiness regime filter (avoid choppy markets)
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr = np.zeros_like(close_arr)
        tr = np.zeros_like(close_arr)
        for i in range(1, len(close_arr)):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]), 
                       abs(low_arr[i] - close_arr[i-1]))
        # Wilder's smoothing
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(close_arr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        # Chop calculation
        chop = np.full_like(close_arr, 50.0, dtype=float)
        for i in range(period, len(close_arr)):
            highest_high = np.max(high_arr[i-period+1:i+1])
            lowest_low = np.min(low_arr[i-period+1:i+1])
            if highest_high != lowest_low:
                chop[i] = 100 * np.log10(sum(tr[i-period+1:i+1]) / 
                                        np.log10(highest_high - lowest_low)) / np.log10(period)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    chop_filter = chop < 61.8  # Trending market
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 30)
    
    for i in range(start_idx, n):
        if np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ma30[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above R1, above EMA34, volume confirmation, trending market
            if close[i] > r1_1d_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below S1, below EMA34, volume confirmation, trending market
            elif close[i] < s1_1d_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume_filter[i] and chop_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Close below S1 or below EMA34
            if close[i] < s1_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Close above R1 or above EMA34
            if close[i] > r1_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals