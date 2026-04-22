#!/usr/bin/env python3

"""
12H Camarilla Pivot Breakout with 1D Trend Filter and Volume Confirmation
Breakout from Camarilla R3/S3 levels on 12h timeframe, filtered by 1d EMA trend and volume spike.
Exits at Camarilla H4/L4 or trend reversal.
Designed for 12-37 trades/year with tight entry criteria to minimize fee drag.
Works in both bull and bear markets by following the 1d trend.
"""

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
    
    # Calculate Camarilla pivot levels on 12h timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    # Daily high, low, close from previous period
    ph = high_series.shift(1)  # Previous high
    pl = low_series.shift(1)   # Previous low
    pc = close_series.shift(1) # Previous close
    
    # Calculate pivot and ranges
    pivot = (ph + pl + pc) / 3
    range_val = ph - pl
    
    # Camarilla levels
    r3 = pivot + (range_val * 1.1 / 2)
    s3 = pivot - (range_val * 1.1 / 2)
    h4 = pivot + (range_val * 1.1)
    l4 = pivot - (range_val * 1.1)
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 34-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):
        # Skip if data not ready
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(h4[i]) or np.isnan(l4[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R3 + 1d uptrend + volume spike
            if close[i] > r3[i] and ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 + 1d downtrend + volume spike
            elif close[i] < s3[i] and ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price reaches H4/L4 or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price reaches H4 or 1d trend turns down
                if close[i] >= h4[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reaches L4 or 1d trend turns up
                if close[i] <= l4[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0