#!/usr/bin/env python3
# 4H_CAMARILLA_R1_S1_BREAKOUT_1D_EMA100_TREND_V2
# Hypothesis: Use Camarilla R1/S1 levels from daily chart with tighter volume and trend filters.
# Long when price breaks above R1 with volume > 2x average and price > EMA100.
# Short when price breaks below S1 with volume > 2x average and price < EMA100.
# Exit when price retests the broken level or trend fails.
# Designed for fewer trades (~20-40/year) to reduce fee drag while maintaining edge in bull/bear markets.

name = "4H_CAMARILLA_R1_S1_BREAKOUT_1D_EMA100_TREND_V2"
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
    
    # 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1 levels from previous 1d bar
    camarilla_r1 = np.full(len(close_1d), np.nan)
    camarilla_s1 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        range_val = ph - pl
        
        camarilla_r1[i] = pc + range_val * 1.1 / 6
        camarilla_s1[i] = pc - range_val * 1.1 / 6
    
    # EMA100 for 1d trend filter
    ema100 = pd.Series(close_1d).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 2 * vol_ma
    
    # Align all 1d data to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema100_aligned = align_htf_to_ltf(prices, df_1d, ema100)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if any critical data is not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema100_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R1 with volume filter in uptrend
            if (high[i] > camarilla_r1_aligned[i] and 
                volume_filter[i] and 
                close[i] > ema100_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S1 with volume filter in downtrend
            elif (low[i] < camarilla_s1_aligned[i] and 
                  volume_filter[i] and 
                  close[i] < ema100_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price falls back below R1 or trend fails
            if (close[i] < camarilla_r1_aligned[i] or 
                close[i] < ema100_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price rises back above S1 or trend fails
            if (close[i] > camarilla_s1_aligned[i] or 
                close[i] > ema100_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals