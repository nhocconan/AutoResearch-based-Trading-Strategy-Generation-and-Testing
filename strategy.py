#!/usr/bin/env python3
"""
4h_1D_Camarilla_Pivot_Breakout_With_Volume_Filter_v13
Hypothesis: 4h price breaks above/below daily Camarilla R4/S4 levels with daily volume > 1.8x 20-period average.
Long when price breaks above R4 + volume condition.
Short when price breaks below S4 + volume condition.
Exit when price crosses daily pivot point (PP).
Focus on tight entry conditions to reduce trade frequency and avoid fee drag. Designed for 4h timeframe to target 25-35 trades/year with strong trend capture in both bull and bear markets.
"""

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
    
    # Daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's values for today's calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Daily VWAP calculation (approximation using typical price)
    typical_price = (high_1d + low_1d + close_1d) / 3
    vwap_numerator = np.cumsum(typical_price * vol_1d)
    vwap_denominator = np.cumsum(vol_1d)
    vwap = np.where(vwap_denominator != 0, vwap_numerator / vwap_denominator, typical_price)
    
    # Camarilla calculation
    range_1d = prev_high - prev_low
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r4 = camarilla_pp + (range_1d * 1.1 / 2)
    camarilla_s4 = camarilla_pp - (range_1d * 1.1 / 2)
    
    # Align 1d data to 4h
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean()
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vwap_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 1d volume > 1.8x 20-period average
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_condition = vol_1d_aligned[i] > (vol_ma_20_aligned[i] * 1.8)
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_r4_aligned[i]
        short_breakout = close[i] < camarilla_s4_aligned[i]
        
        # Exit condition
        long_exit = close[i] < camarilla_pp_aligned[i]
        short_exit = close[i] > camarilla_pp_aligned[i]
        
        if position == 0:
            if long_breakout and vol_condition:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_condition:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1D_Camarilla_Pivot_Breakout_With_Volume_Filter_v13"
timeframe = "4h"
leverage = 1.0