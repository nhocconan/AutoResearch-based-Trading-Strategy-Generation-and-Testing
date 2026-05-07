#!/usr/bin/env python3
# 4h_CamillaR1_S1_Breakout_1dTrend_Volume
# Hypothesis: Camarilla R1/S1 levels from daily chart act as key support/resistance.
# Breakouts above R1 or below S1 with volume confirmation and 1-day trend filter capture
# institutional breakout moves. Works in both bull and bear markets by trading with
# the higher timeframe trend. Uses tight entry conditions to limit trades and reduce fee drag.

name = "4h_CamillaR1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) == 0:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels from previous day
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    # Using previous day's values (shifted by 1)
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # First day will have invalid values (from roll) but will be filtered by min_periods later
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 12
    s1 = prev_close - 1.1 * rang / 12
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate EMA34 for 1-day trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike detection: 2.0x average volume (40-period = ~3.33 days on 4h chart)
    vol_ma = pd.Series(volume).rolling(window=40, min_periods=40).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 40)  # Need EMA34 and volume MA data, plus Camarilla needs prev day
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN or invalid
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, price above EMA34 (uptrend), volume spike
            if (high[i] > r1_aligned[i] and 
                close[i] > ema34_1d_aligned[i] and 
                volume[i] > 2.0 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, price below EMA34 (downtrend), volume spike
            elif (low[i] < s1_aligned[i] and 
                  close[i] < ema34_1d_aligned[i] and 
                  volume[i] > 2.0 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below S1 OR price crosses below EMA34
            if (low[i] < s1_aligned[i] or close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above R1 OR price crosses above EMA34
            if (high[i] > r1_aligned[i] or close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals