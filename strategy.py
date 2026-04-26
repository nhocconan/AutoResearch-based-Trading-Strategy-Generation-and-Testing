#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dEMA34_TrendFilter_v3
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
Uses tighter volume spike threshold (3.0x) and requires close beyond Camarilla level for confirmation.
Reduces trade frequency to avoid fee drag while maintaining edge in bull/bear via 1d trend alignment.
Target: 12-30 trades/year on 12h timeframe.
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
    
    # Get 1d data for trend and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate previous day's Camarilla pivot levels (R1, S1)
    high_1d_prev = np.roll(df_1d['high'].values, 1)
    low_1d_prev = np.roll(df_1d['low'].values, 1)
    close_1d_prev = np.roll(df_1d['close'].values, 1)
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    close_1d_prev[0] = np.nan
    
    pivot = (high_1d_prev + low_1d_prev + close_1d_prev) / 3.0
    range_1d = high_1d_prev - low_1d_prev
    r1 = pivot + (range_1d * 1.0 / 12.0)
    s1 = pivot - (range_1d * 1.0 / 12.0)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 3.0x 20-period MA (tighter than before)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 3.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price closes above R1 with 1d bullish trend and volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price closes below S1 with 1d bearish trend and volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below S1 OR 1d trend turns bearish
            if (close[i] < s1_aligned[i] or close[i] < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above R1 OR 1d trend turns bullish
            if (close[i] > r1_aligned[i] or close[i] > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dEMA34_TrendFilter_v3"
timeframe = "12h"
leverage = 1.0