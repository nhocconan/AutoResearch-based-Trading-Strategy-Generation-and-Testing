#!/usr/bin/env python3
"""
1d_Weekly_Pivot_Reversal_With_Volume_Filter
Hypothesis: Weekly pivot levels (R1/S1) act as strong support/resistance. Price approaching these levels with volume confirmation and weekly trend alignment offers high-probability reversals. Uses 1d timeframe to reduce trade frequency and capture multi-day moves. Weekly trend filter ensures trades align with higher-timeframe momentum. Designed to work in both bull and bear markets by fading extremes in ranging markets and continuing trends in trending markets.
"""

name = "1d_Weekly_Pivot_Reversal_With_Volume_Filter"
timeframe = "1d"
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
    
    # Get weekly data for pivot points and trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points: (H+L+C)/3
    weekly_pp = (df_weekly['high'] + df_weekly['low'] + df_weekly['close']) / 3
    weekly_r1 = 2 * weekly_pp - df_weekly['low']
    weekly_s1 = 2 * weekly_pp - df_weekly['high']
    
    # Align weekly pivot levels to daily timeframe
    pp_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1.values)
    
    # Weekly trend filter: EMA(50) on weekly close
    ema50_weekly = pd.Series(df_weekly['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Volume confirmation: current volume > 1.8x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price touches S1 with volume confirmation and above weekly EMA50 (bullish bias)
            if (low[i] <= s1_aligned[i] * 1.001 and  # Allow small buffer for wick
                volume_filter[i] and
                close[i] > ema50_weekly_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches R1 with volume confirmation and below weekly EMA50 (bearish bias)
            elif (high[i] >= r1_aligned[i] * 0.999 and  # Allow small buffer for wick
                  volume_filter[i] and
                  close[i] < ema50_weekly_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches PP or weekly trend turns bearish
            if (close[i] >= pp_aligned[i] * 0.999 or  # Reached pivot point
                close[i] < ema50_weekly_aligned[i]):  # Weekly trend turned bearish
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches PP or weekly trend turns bullish
            if (close[i] <= pp_aligned[i] * 1.001 or  # Reached pivot point
                close[i] > ema50_weekly_aligned[i]):  # Weekly trend turned bullish
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals