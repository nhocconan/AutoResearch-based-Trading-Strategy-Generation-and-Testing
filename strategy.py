#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Breakout_VolumeFilter_V1
Hypothesis: Camarilla pivot R1/S1 breakout on daily timeframe with volume confirmation (>1.5x 20-day average) works for BTC and ETH in both bull and bear markets. Uses 1-week timeframe for trend filter (price above/below 21-period EMA) to avoid false breakouts. Target: 15-25 trades/year per symbol (60-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load weekly data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_prev = high_prev - low_prev
    
    # Camarilla R1, R2, S1, S2
    r1 = close_prev + range_prev * 1.1 / 12.0
    r2 = close_prev + range_prev * 1.1 / 6.0
    s1 = close_prev - range_prev * 1.1 / 12.0
    s2 = close_prev - range_prev * 1.1 / 6.0
    
    # Align Camarilla levels to intraday timeframe (using open_time for alignment)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Weekly trend filter: 21-period EMA
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume confirmation: 20-day average
    vol_ma_20 = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>1.5x average)
        volume_ok = volume > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume and weekly uptrend
            if price > r1_aligned[i] and price > ema_21_1w_aligned[i]:
                if volume_ok:
                    signals[i] = 0.25
                    position = 1
            # Short: price breaks below S1 with volume and weekly downtrend
            elif price < s1_aligned[i] and price < ema_21_1w_aligned[i]:
                if volume_ok:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price drops below pivot or weekly trend turns down
            if price < pivot[i] or price < ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price rises above pivot or weekly trend turns up
            if price > pivot[i] or price > ema_21_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_Pivot_Breakout_VolumeFilter_V1"
timeframe = "1d"
leverage = 1.0