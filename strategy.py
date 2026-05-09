#!/usr/bin/env python3
# Hypothesis: 4h Camarilla pivot breakout with 1d EMA trend filter and volume confirmation.
# Uses Camarilla levels (R1/S1) from daily pivot for precise entry/exit.
# Long when price breaks above R1 with 1d EMA(34) uptrend and volume > 1.5x average.
# Short when price breaks below S1 with 1d EMA(34) downtrend and volume > 1.5x average.
# Exits when price returns to daily pivot or trend reverses.
# Target: 80-160 total trades over 4 years (20-40/year) with size 0.25.
# Works in bull/bear via trend filter and volatility-based exits.

name = "4h_Camarilla_R1S1_EMA34_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1-day EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close']
    ema_34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily pivot and Camarilla levels (R1, S1)
    high_1d = df_1d['high']
    low_1d = df_1d['low']
    close_1d = df_1d['close']
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.0833)  # R1 = C + ((H-L) * 1.0833)
    s1 = pivot - (range_1d * 1.0833)  # S1 = C - ((H-L) * 1.0833)
    
    r1_values = r1.values
    s1_values = s1.values
    pivot_values = pivot.values
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_values)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pivot_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > R1, EMA uptrend, volume confirmation
            if close[i] > r1_aligned[i] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price < S1, EMA downtrend, volume confirmation
            elif close[i] < s1_aligned[i] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to pivot OR EMA trend turns down
            if close[i] <= pivot_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to pivot OR EMA trend turns up
            if close[i] >= pivot_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals