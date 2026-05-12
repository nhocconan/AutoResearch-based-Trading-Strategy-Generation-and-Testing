#!/usr/bin/env python3
"""
12H_CAMARILLA_R3_S3_BREAKOUT_1D_WEEKLY_TREND_FILTER
Hypothesis: On 12h timeframe, take long when price breaks above 1-day Camarilla R3 with volume spike and weekly trend up (EMA20 > EMA50), short when breaks below S3 with volume spike and weekly trend down (EMA20 < EMA50). Exit when price crosses opposite S3/R3. Uses weekly EMA trend filter to avoid counter-trend trades, reducing whipsaw in ranging markets. Designed for ~15-25 trades/year on 12h to minimize fee drag.
"""
name = "12H_CAMARILLA_R3_S3_BREAKOUT_1D_WEEKLY_TREND_FILTER"
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
    
    # Calculate Camarilla levels from previous day
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    
    for i in range(1, n):
        camarilla_r3[i] = close[i-1] + (high[i-1] - low[i-1]) * 1.1 / 2
        camarilla_s3[i] = close[i-1] - (high[i-1] - low[i-1]) * 1.1 / 2
    
    # 1d data for volume spike confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma_20d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = vol_1d / vol_ma_20d  # Current volume / 20-day average
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Weekly EMA trend filter (EMA20 > EMA50 = uptrend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema20_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(vol_spike_aligned[i]) or np.isnan(ema20_aligned[i]) or np.isnan(ema50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with volume spike and weekly uptrend
            if (high[i] > camarilla_r3[i] and 
                vol_spike_aligned[i] > 1.5 and
                ema20_aligned[i] > ema50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume spike and weekly downtrend
            elif (low[i] < camarilla_s3[i] and 
                  vol_spike_aligned[i] > 1.5 and
                  ema20_aligned[i] < ema50_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below S3 (reversion to mean)
            if close[i] < camarilla_s3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above R3 (reversion to mean)
            if close[i] > camarilla_r3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals