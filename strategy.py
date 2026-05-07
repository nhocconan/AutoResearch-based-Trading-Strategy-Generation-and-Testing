#!/usr/bin/env python3
# 4h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Volume_v2
# Hypothesis: Refined version focusing on tighter entry conditions to reduce trade frequency.
# Uses Camarilla R1/S1 from 1d, requires close to close beyond the level (not just touch),
# filters by 1d EMA34 trend and volume spike (>2x 20-period average) on 4h.
# Uses hysteresis in exits: requires close to revert back inside the level to exit.
# Designed for 4h to achieve 20-40 trades/year per symbol, targeting 80-160 total over 4 years.
# Works in bull/bear via trend-following breakouts with volume confirmation.

name = "4h_Camarilla_Pivot_R1S1_Breakout_1dTrend_Volume_v2"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels: R1, S1
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_1d_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume spike on 4h timeframe (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(ema_34_1d_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > R1 + above 1d EMA34 trend + volume spike
            if close[i] > r1_4h[i] and close[i] > ema_34_1d_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close < S1 + below 1d EMA34 trend + volume spike
            elif close[i] < s1_4h[i] and close[i] < ema_34_1d_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < R1 (revert inside the level)
            if close[i] < r1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > S1 (revert inside the level)
            if close[i] > s1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals