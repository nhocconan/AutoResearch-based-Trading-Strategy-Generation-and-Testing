#!/usr/bin/env python3
# 12h_1d_Camarilla_R1S1_Breakout_Volume_Spike
# Hypothesis: 12h Camarilla breakout (R1/S1 levels) with 1d volume spike and 1w trend filter.
# Long: price breaks above R1 + volume spike > 1.5x average + 1w close > 1w open (bullish trend)
# Short: price breaks below S1 + volume spike > 1.5x average + 1w close < 1w open (bearish trend)
# Exit: price returns to Camarilla pivot (PP) or opposite signal.
# Uses volume spike to avoid false breakouts and 1w trend to align with higher timeframe momentum.
# Target: 15-30 trades/year per symbol for low fee attrition.

name = "12h_1d_Camarilla_R1S1_Breakout_Volume_Spike"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # first day uses same day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla levels
    pp = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    r1 = pp + (range_ * 1.1 / 12)
    s1 = pp - (range_ * 1.1 / 12)
    r2 = pp + (range_ * 1.1 / 6)
    s2 = pp - (range_ * 1.1 / 6)
    
    # Align 1d Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1w trend: bullish if close > open, bearish if close < open
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    trend_1w = close_1w - open_1w  # positive = bullish, negative = bearish
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # Calculate volume spike: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (vol_ma.values * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(trend_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for breakout with volume spike and trend alignment
            # Long: price breaks above R1 + volume spike + bullish 1w trend
            if (close[i] > r1_aligned[i] and vol_spike[i] and 
                trend_1w_aligned[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume spike + bearish 1w trend
            elif (close[i] < s1_aligned[i] and vol_spike[i] and 
                  trend_1w_aligned[i] < 0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: exit if price returns to PP or opposite breakout occurs
            if close[i] <= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: exit if price returns to PP or opposite breakout occurs
            if close[i] >= pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals