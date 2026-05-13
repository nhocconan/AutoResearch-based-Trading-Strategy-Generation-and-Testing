#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Spike
Hypothesis: Use daily Camarilla pivot levels (R1/S1) for breakout entries, filtered by daily trend (close > EMA34) and volume spikes (volume > 2x volume EMA34). Exit on opposite breakout or trend reversal. Camarilla levels provide high-probability support/resistance; volume confirms institutional interest; trend filter ensures alignment with higher timeframe direction. Designed for 4h to balance trade frequency and avoid fee drag, targeting 20-50 trades/year.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots, trend filter, and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous day's range
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Using previous day's high, low, close to avoid look-ahead
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Calculate volume EMA34 for volume spike filter
    vol_1d = df_1d['volume'].values
    vol_ema_34 = pd.Series(vol_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all daily indicators to 4h timeframe
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    vol_ema_34_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R1 with volume spike and uptrend
            if (high[i] > r1_aligned[i] and 
                volume[i] > 2 * vol_ema_34_aligned[i] and 
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume spike and downtrend
            elif (low[i] < s1_aligned[i] and 
                  volume[i] > 2 * vol_ema_34_aligned[i] and 
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below S1 or trend reversal
            if low[i] < s1_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above R1 or trend reversal
            if high[i] > r1_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals