#!/usr/bin/env python3
"""
1d_1w_Camarilla_Pivot_Breakout_With_Volume_Confirmation
Hypothesis: Daily close above/below weekly Camarilla levels with volume confirmation.
Long when daily close > weekly R4 + daily volume > 2x 20-day avg volume + weekly close > weekly SMA50.
Short when daily close < weekly S4 + daily volume > 2x 20-day avg volume + weekly close < weekly SMA50.
Exit when daily close crosses weekly pivot point (PP) or weekly trend reverses.
Designed for daily timeframe to target 5-15 trades/year with strong trend capture in both bull/bear markets.
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
    volume = prices['volume'].values
    
    # Weekly Camarilla pivot levels (based on previous week)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's values for current week calculation
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = high_1w[0]
    prev_low[0] = low_1w[0]
    prev_close[0] = close_1w[0]
    
    # Camarilla calculation
    range_1w = prev_high - prev_low
    camarilla_pp = (prev_high + prev_low + prev_close) / 3
    camarilla_r4 = camarilla_pp + (range_1w * 1.1 / 2)
    camarilla_s4 = camarilla_pp - (range_1w * 1.1 / 2)
    
    # Align weekly Camarilla levels to daily
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pp)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Daily volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Weekly trend filter
    close_1w = df_1w['close'].values
    sma_50 = pd.Series(close_1w).rolling(window=50, min_periods=50).mean()
    sma_50_aligned = align_htf_to_ltf(prices, df_1w, sma_50.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(sma_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: daily volume > 2x 20-day average
        vol_condition = volume[i] > (vol_ma_20[i] * 2.0)
        
        # Weekly trend condition
        uptrend = close[i] > sma_50_aligned[i]
        downtrend = close[i] < sma_50_aligned[i]
        
        # Breakout conditions (using daily close)
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

name = "1d_1w_Camarilla_Pivot_Breakout_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0