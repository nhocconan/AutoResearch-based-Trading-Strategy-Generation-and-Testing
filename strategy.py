#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Breakout_Volume_Trend
Hypothesis: Weekly pivot points provide strong institutional support/resistance. 
Breakouts above R1 or below S1 with volume confirmation and aligned daily trend 
capture major moves in both bull and bear markets. Weekly timeframe reduces noise, 
while 6h timeframe captures timely entries with lower trade frequency.
"""

name = "6h_Weekly_Pivot_Breakout_Volume_Trend"
timeframe = "6h"
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
    
    # Calculate weekly pivot points (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    
    # Prior week's high, low, close
    ph = df_1w['high'].values
    pl = df_1w['low'].values
    pc = df_1w['close'].values
    
    # Pivot point and support/resistance levels
    pp = (ph + pl + pc) / 3.0
    r1 = 2 * pp - pl
    s1 = 2 * pp - ph
    r2 = pp + (ph - pl)
    s2 = pp - (ph - pl)
    
    # Align weekly levels to 6h timeframe (values update only after weekly close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Daily trend filter: EMA 50 on 1d
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if position == 0:
            # LONG: Break above R1 with volume confirmation and uptrend
            if close[i] > r1_aligned[i] and volume_confirm[i]:
                if close[i] > ema_50_1d_aligned[i]:  # Daily uptrend filter
                    signals[i] = 0.25
                    position = 1
            # SHORT: Break below S1 with volume confirmation and downtrend
            elif close[i] < s1_aligned[i] and volume_confirm[i]:
                if close[i] < ema_50_1d_aligned[i]:  # Daily downtrend filter
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below pivot point or R2 break fails
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above pivot point or S2 break fails
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals