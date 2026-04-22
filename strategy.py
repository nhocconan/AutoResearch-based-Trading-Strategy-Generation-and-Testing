#!/usr/bin/env python3

"""
Hypothesis: 12-hour chart with daily pivot points and volume confirmation.
Trade long when price breaks above daily R1 pivot with volume confirmation and daily trend up.
Trade short when price breaks below daily S1 pivot with volume confirmation and daily trend down.
Uses pivot points as key support/resistance levels with volume confirmation to filter false breakouts.
Designed for low trade frequency (12-37 trades/year) by requiring multiple confirmations: 
pivot breakout, volume spike, and trend alignment. Works in both bull and bear markets by 
following the daily trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for pivot points and trend - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard formula)
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    pivot = (daily_high + daily_low + daily_close) / 3.0
    r1 = 2 * pivot - daily_low
    s1 = 2 * pivot - daily_high
    
    # Align pivot levels to 12h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    
    # Daily EMA34 for trend direction
    ema34_daily = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema34_daily_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.8 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above R1 + daily uptrend + volume spike
            if close[i] > r1_aligned[i] and ema34_daily_aligned[i] > ema34_daily_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + daily downtrend + volume spike
            elif close[i] < s1_aligned[i] and ema34_daily_aligned[i] < ema34_daily_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to pivot level or opposite pivot level touched
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to pivot or breaks below S1
                if close[i] <= pivot_aligned[i] or close[i] < s1_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to pivot or breaks above R1
                if close[i] >= pivot_aligned[i] or close[i] > r1_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Pivot_R1S1_Breakout_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0