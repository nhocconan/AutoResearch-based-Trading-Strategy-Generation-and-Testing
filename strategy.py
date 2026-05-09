#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_BullBearPower_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray calculation (13-period EMA)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 13-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Bull Power and Bear Power on 1d
    bull_power_1d = high_1d - ema13_1d if 'high_1d' in locals() else df_1d['high'].values - ema13_1d
    bear_power_1d = low_1d - ema13_1d if 'low_1d' in locals() else df_1d['low'].values - ema13_1d
    
    # Actually compute high and low arrays for 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Get 1w data for trend filter (40-period EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    
    # Align 1d indicators to 6h
    bull_power_1d_6h = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_6h = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Align 1w EMA to 6h
    ema40_1w_6h = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Volume filter: current 6h volume > 1.5 * 20-period average (less strict)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 40)  # Need enough data for volume MA and EMAs
    
    for i in range(start_idx, n):
        if (np.isnan(bull_power_1d_6h[i]) or np.isnan(bear_power_1d_6h[i]) or
            np.isnan(ema40_1w_6h[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bull_power = bull_power_1d_6h[i]
        bear_power = bear_power_1d_6h[i]
        trend = ema40_1w_6h[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: bull power positive AND price above weekly trend AND volume
            if bull_power > 0 and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: bear power negative AND price below weekly trend AND volume
            elif bear_power < 0 and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: bull power turns negative (momentum loss)
            if bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: bear power turns positive (momentum loss)
            if bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals