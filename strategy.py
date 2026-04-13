#!/usr/bin/env python3
"""
12h_1d1w_Camarilla_Pivot_Breakout_With_Volume_Confirmation_v3
Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter.
Long when price breaks above 12h resistance R4 + 1d volume > 1.8x 20-period average + 1w close > 1w SMA50.
Short when price breaks below 12h support S4 + 1d volume > 1.8x 20-period average + 1w close < 1w SMA50.
Exit when price crosses 12h pivot point (PP) or 1w trend reverses.
Designed for 12h timeframe to target 15-30 trades/year with strong trend capture in both bull/bear markets.
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
    
    # 12h Camarilla pivot levels (based on previous day)
    # For 12h timeframe, we use daily OHLC from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for today's calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla calculation
    range_1d = prev_high - prev_low
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r4 = camarilla_pp + (range_1d * 1.1 / 2)
    camarilla_s4 = camarilla_pp - (range_1d * 1.1 / 2)
    
    # Align 1d Camarilla levels to 12h
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 1d volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean()
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20.values)
    
    # 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    sma_50 = pd.Series(close_1w).rolling(window=50, min_periods=50).mean()
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(sma_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 1d volume > 1.8x 20-period average
        vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
        vol_condition = vol_1d_aligned[i] > (vol_ma_20_aligned[i] * 1.8)
        
        # 1w trend condition
        uptrend = close[i] > sma_50_aligned[i]
        downtrend = close[i] < sma_50_aligned[i]
        
        # Breakout conditions
        long_breakout = close[i] > camarilla_r4_aligned[i]
        short_breakout = close[i] < camarilla_s4_aligned[i]
        
        # Exit conditions
        long_exit = close[i] < camarilla_pp_aligned[i]
        short_exit = close[i] > camarilla_pp_aligned[i]
        trend_reverse_long = close[i] < sma_50_aligned[i]  # uptrend broken
        trend_reverse_short = close[i] > sma_50_aligned[i]  # downtrend broken
        
        if position == 0:
            if long_breakout and vol_condition and uptrend:
                position = 1
                signals[i] = position_size
            elif short_breakout and vol_condition and downtrend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit or trend_reverse_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            if short_exit or trend_reverse_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d1w_Camarilla_Pivot_Breakout_With_Volume_Confirmation_v3"
timeframe = "12h"
leverage = 1.0