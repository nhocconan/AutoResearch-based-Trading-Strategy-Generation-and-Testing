#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike
Hypothesis: Camarilla pivot levels (R1/S1) from daily pivot act as strong support/resistance.
Breakouts above R1 or below S1 with volume confirmation and trend filter (EMA34 on 1d) capture
trends in both bull and bear markets. Low trade frequency (target 20-40/year) with clear
entry/exit rules minimizes fee drag. Works in bull via breakouts, in bear via breakdowns.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike"
timeframe = "4h"
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
    
    # Calculate daily pivot points from 1d data
    df_1d = get_htf_data(prices, '1d')
    # Typical price for each day
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Pivot point
    pivot = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Calculate support and resistance levels
    # R1 = (2 * Pivot) - Low
    # S1 = (2 * Pivot) - High
    r1 = (2 * pivot) - df_1d['low']
    s1 = (2 * pivot) - df_1d['high']
    
    # Align to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot.values)
    
    # 1-day EMA34 trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA warmup
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and uptrend
            if close[i] > r1_aligned[i] and volume_confirm[i]:
                # Additional filter: only take long if price above 1-day EMA34 (uptrend)
                if close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price breaks below S1 with volume confirmation and downtrend
            elif close[i] < s1_aligned[i] and volume_confirm[i]:
                # Additional filter: only take short if price below 1-day EMA34 (downtrend)
                if close[i] < ema34_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below pivot or S1 (support break)
            if close[i] < s1_aligned[i] or close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above pivot or R1 (resistance break)
            if close[i] > r1_aligned[i] or close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals