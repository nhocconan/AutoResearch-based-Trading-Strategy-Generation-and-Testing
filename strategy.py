#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe act as strong support/resistance.
Breakouts above R1 or below S1 with volume confirmation and aligned 1d trend (close > EMA34) signal continuation.
Uses 25% position size to limit trade frequency (~15-30/year) and minimize fee drag in 12-hour bars.
Works in both bull and bear markets by requiring trend alignment and volume confirmation.
"""

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get 1d data for Camarilla pivots and trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels (R1, S1) from previous day
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C, H, L are from previous completed day
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (wait for previous day's close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d trend filter: EMA(34) on close
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup
        if position == 0:
            # LONG: Close crosses above R1, volume confirmation, price above 1d EMA34 (uptrend)
            if (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1] and 
                volume_filter[i] and 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Close crosses below S1, volume confirmation, price below 1d EMA34 (downtrend)
            elif (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1] and 
                  volume_filter[i] and 
                  close[i] < ema34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close crosses below S1 (opposite level) OR volume drops
            if (close[i] < s1_aligned[i] and close[i-1] >= s1_aligned[i-1]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close crosses above R1 (opposite level) OR volume drops
            if (close[i] > r1_aligned[i] and close[i-1] <= r1_aligned[i-1]) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals