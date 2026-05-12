#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_R1_S1_Breakout_1dTrend
Hypothesis: Camarilla pivot levels (R1/S1) from daily timeframe with trend filter on 12h.
Enters long when price breaks above R1 with daily trend up, short when breaks below S1 with daily trend down.
Uses 12h timeframe for low trade frequency (12-37/year) to minimize fee drag. Designed to work in both bull and bear markets via trend filter.
"""

name = "12h_Camarilla_Pivot_R1_S1_Breakout_1dTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period"""
    # Pivot point
    pivot = (high + low + close) / 3
    # Range
    range_val = high - low
    # Camarilla levels
    r1 = close + (range_val * 1.1 / 12)
    s1 = close - (range_val * 1.1 / 12)
    return pivot, r1, s1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get daily data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)

    # Calculate Camarilla levels on daily data
    _, r1_1d, s1_1d = calculate_camarilla(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values)
    
    # Calculate daily EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily levels to 12h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 12h ATR for volatility filter
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr1])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(34, n):  # Start after EMA warmup
        # Get aligned values for current 12h bar
        r1_val = r1_1d_aligned[i]
        s1_val = s1_1d_aligned[i]
        ema34_val = ema34_1d_aligned[i]
        atr_val = atr[i]
        
        # Skip if any required data is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(ema34_val) or np.isnan(atr_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price breaks above R1 + daily uptrend + volatility filter
            if (close[i] > r1_val and 
                close[i] > ema34_val and
                atr_val > 0):  # Ensure volatility exists
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + daily downtrend + volatility filter
            elif (close[i] < s1_val and 
                  close[i] < ema34_val and
                  atr_val > 0):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below R1 or trend changes
            if (close[i] < r1_val or close[i] < ema34_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters above S1 or trend changes
            if (close[i] > s1_val or close[i] > ema34_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals