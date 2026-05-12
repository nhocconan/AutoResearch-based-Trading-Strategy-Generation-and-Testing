#!/usr/bin/env python3
"""
4H_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Combine Camarilla R1/S1 breakout from 4h with 1d EMA trend and volume confirmation.
Go long when price breaks above R1 with price above 1d EMA34 and volume spike.
Go short when price breaks below S1 with price below 1d EMA34 and volume spike.
Exit when price returns to Camarilla Pivot point or trend changes.
Camarilla levels provide institutional support/resistance; EMA34 filters trend; volume confirms breakout strength.
Designed for 15-25 trades/year to avoid fee drag while capturing strong moves.
"""

name = "4H_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for the day"""
    # Pivot point
    pivot = (high + low + close) / 3
    # Range
    range_val = high - low
    # Camarilla levels
    r1 = close + (range_val * 1.1 / 12)
    s1 = close - (range_val * 1.1 / 12)
    r2 = close + (range_val * 1.1 / 6)
    s2 = close - (range_val * 1.1 / 6)
    r3 = close + (range_val * 1.1 / 4)
    s3 = close - (range_val * 1.1 / 4)
    r4 = close + (range_val * 1.1 / 2)
    s4 = close - (range_val * 1.1 / 2)
    return pivot, r1, s1, r2, s2, r3, s3, r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get 1d data for EMA trend filter ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)

    # Calculate EMA34 on daily close
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)

    # Calculate Camarilla levels from 1d data for 4h breakout
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot_1d, r1_1d, s1_1d, r2_1d, s2_1d, r3_1d, s3_1d, r4_1d, s4_1d = calculate_camarilla(
        high_1d, low_1d, close_1d
    )
    
    # Align Camarilla levels to 4h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)

    # Volume confirmation: current volume > 1.5x average of last 6 periods
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    volume_ok = volume > (1.5 * vol_ma)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(20, n):  # Start after warmup
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(pivot_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or 
            np.isnan(s1_1d_aligned[i]) or
            np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: price breaks above R1 + price above 1d EMA34 + volume spike
            if close[i] > r1_1d_aligned[i] and close[i] > ema_34_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below S1 + price below 1d EMA34 + volume spike
            elif close[i] < s1_1d_aligned[i] and close[i] < ema_34_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price returns to pivot or falls below EMA34
            if close[i] <= pivot_1d_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price returns to pivot or rises above EMA34
            if close[i] >= pivot_1d_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals