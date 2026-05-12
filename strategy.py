#!/usr/bin/env python3
# 12H_VOLATILITY_BREAKOUT_1D_TREND_FILTER
# Hypothesis: Breakouts from 12h volatility-based channels (ATR-based) with daily trend filter capture momentum in both bull and bear markets.
# In bull markets: buy breakouts above upper channel in uptrend. In bear markets: sell breakdowns below lower channel in downtrend.
# Uses volatility contraction/expansion to filter low-volatility periods and avoid whipsaws.
# Target: 12-37 trades/year on 12h timeframe (50-150 total over 4 years).

name = "12H_VOLATILITY_BREAKOUT_1D_TREND_FILTER"
timeframe = "12h"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 12h ATR-based volatility channel (20-period)
    atr_period = 20
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # First TR
    atr = np.zeros_like(close)
    atr[0] = tr[0]
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Calculate upper and lower channels (ATR multiplier = 1.5)
    mult = 1.5
    upper_channel = np.roll(close, 1) + atr * mult
    lower_channel = np.roll(close, 1) - atr * mult
    # First value has no previous close, set to extreme values
    upper_channel[0] = high[0] + atr[0] * mult
    lower_channel[0] = low[0] - atr[0] * mult
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one period of data
    
    for i in range(start_idx, n):
        # Skip if trend filter data is not ready
        if np.isnan(ema34_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper channel in uptrend
            if (close[i] > upper_channel[i] and 
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower channel in downtrend
            elif (close[i] < lower_channel[i] and 
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls below lower channel or trend reversal
            if (close[i] < lower_channel[i] or 
                close[i] <= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises above upper channel or trend reversal
            if (close[i] > upper_channel[i] or 
                close[i] >= ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals