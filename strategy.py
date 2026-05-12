#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_1D_TREND_FILTER
# Hypothesis: Camarilla R1/S1 levels from the daily chart act as significant support/resistance levels.
# Breaking above R1 with the daily trend up (price > EMA50) signals a long entry.
# Breaking below S1 with the daily trend down (price < EMA50) signals a short entry.
# The trend filter reduces false breakouts in choppy markets, improving win rate.
# Works in both bull and bear markets by following the daily trend direction.
# Target: 20-40 trades per year to stay under the 400 total trade limit over 4 years.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_1D_TREND_FILTER"
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
    
    # Get 1D data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R1 and S1 levels from the previous day's data
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        ph = high_1d[i-1]  # Previous day high
        pl = low_1d[i-1]   # Previous day low
        pc = close_1d[i-1] # Previous day close
        range_val = ph - pl
        
        camarilla_r1[i] = pc + range_val * 1.1 / 6
        camarilla_s1[i] = pc - range_val * 1.1 / 6
    
    # Daily EMA50 for trend filter
    ema50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1D data to 4H timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any critical data is not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R1 with daily uptrend
            if (high[i] > camarilla_r1_aligned[i] and 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with daily downtrend
            elif (low[i] < camarilla_s1_aligned[i] and 
                  close[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below R1 or trend turns down
            if (close[i] < camarilla_r1_aligned[i] or 
                close[i] < ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above S1 or trend turns up
            if (close[i] > camarilla_s1_aligned[i] or 
                close[i] > ema50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals