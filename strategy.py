#!/usr/bin/env python3
"""
4h_PivotPoint_Breakout_1dTrend_Volume
Hypothesis: Uses daily Camarilla pivot levels (S1, R1) for breakout entries, filtered by 1-day EMA34 trend and volume confirmation. Exits when price crosses the daily pivot point or reverses trend. Designed for low trade frequency (20-40/year) with clear structure-based entries that work in both bull and bear markets by following the higher timeframe trend.
"""

name = "4h_PivotPoint_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for pivot points and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot points from previous day
    # Using standard formula: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # But we only need S1 and R1 for entries
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    r1 = pivot + range_hl * 1.1 / 4  # Camarilla R1
    s1 = pivot - range_hl * 1.1 / 4  # Camarilla S1
    
    # Align to 4h timeframe (these levels are valid for the entire day)
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    pivot_4h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Daily trend filter (EMA34)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (24-period average on 4h = 6 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(pivot_4h[i]) or np.isnan(ema_34_4h[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: price breaks above R1 + above daily EMA34 + volume
            if (close[i] > r1_4h[i] and 
                close[i] > ema_34_4h[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + below daily EMA34 + volume
            elif (close[i] < s1_4h[i] and 
                  close[i] < ema_34_4h[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to pivot point OR trend turns down
                if (close[i] <= pivot_4h[i]) or \
                   (close[i] < ema_34_4h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to pivot point OR trend turns up
                if (close[i] >= pivot_4h[i]) or \
                   (close[i] > ema_34_4h[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals