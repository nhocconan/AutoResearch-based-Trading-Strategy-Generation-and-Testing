#!/usr/bin/env python3
"""
4h_1d_Camarilla_R1_S1_Breakout_TrendFilter_Volume
Hypothesis: Breakouts above 1d Camarilla R1 or below S1 with 4h trend alignment and volume confirmation work in both bull and bear markets.
The 1d Camarilla levels provide institutional support/resistance. Trend filter ensures trades follow higher timeframe direction.
Volume confirmation reduces false breakouts. Target: 20-40 trades/year to avoid fee drag.
"""

name = "4h_1d_Camarilla_R1_S1_Breakout_TrendFilter_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (using prior day)
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla: Range = high - low
    range_prev = high_prev - low_prev
    # Camarilla levels: Close ± (Range * 1.1/2)
    camarilla_p = close_prev  # Pivot is close of previous day
    camarilla_r1 = camarilla_p + (range_prev * 1.1 / 2)
    camarilla_s1 = camarilla_p - (range_prev * 1.1 / 2)
    
    # Align daily Camarilla levels to 4h timeframe (wait for daily bar to close)
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get 4h price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.3x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need daily Camarilla (needs 1 day), EMA34 (34 bars), volume EMA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_p_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above EMA34 (uptrend) AND price breaks above Camarilla R1 with volume
            if close[i] > ema_34_aligned[i] and high[i] > camarilla_r1_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below EMA34 (downtrend) AND price breaks below Camarilla S1 with volume
            elif close[i] < ema_34_aligned[i] and low[i] < camarilla_s1_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Camarilla pivot P OR trend turns bearish
            if low[i] < camarilla_p_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Camarilla pivot P OR trend turns bullish
            if high[i] > camarilla_p_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals