#!/usr/bin/env python3
"""
12H_CAMARILLA_R3_S3_BREAKOUT_1W_TREND_FILTER
Hypothesis: On 12h timeframe, take long when price breaks above weekly R3 with
above-average weekly volume and weekly close above weekly EMA20 (uptrend).
Take short when price breaks below weekly S3 with above-average weekly volume
and weekly close below weekly EMA20 (downtrend). Exit when price reverts to
opposite S3/R3 level. Uses weekly timeframe for trend and volume to reduce
noise and focus on institutional-grade moves. Designed for ~15-30 trades/year
on 12h to minimize fee drag while capturing strong trending moves.
"""
name = "12H_CAMARILLA_R3_S3_BREAKOUT_1W_TREND_FILTER"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get weekly data for trend and volume context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    weekly_close = df_1w['close'].values
    ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_uptrend = weekly_close > ema20
    weekly_downtrend = weekly_close < ema20
    
    # Calculate weekly volume and its 20-period average for volume spike
    weekly_volume = df_1w['volume'].values
    vol_ma_20w = pd.Series(weekly_volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = weekly_volume / vol_ma_20w  # Current weekly vol / 20-week EMA
    
    # Align weekly indicators to 12h timeframe (properly delayed for weekly close)
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    vol_spike_aligned = align_htf_to_ltf(prices, df_1w, vol_spike)
    
    # Calculate weekly OHLC for Camarilla levels (using prior week's data)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close_prev = df_1w['close'].values
    
    camarilla_r3 = np.full(len(weekly_close), np.nan)
    camarilla_s3 = np.full(len(weekly_close), np.nan)
    
    for i in range(1, len(weekly_close)):
        camarilla_r3[i] = weekly_close_prev[i-1] + (weekly_high[i-1] - weekly_low[i-1]) * 1.1 / 2
        camarilla_s3[i] = weekly_close_prev[i-1] - (weekly_high[i-1] - weekly_low[i-1]) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above weekly R3 with volume spike and weekly uptrend
            if (high[i] > camarilla_r3_aligned[i] and 
                vol_spike_aligned[i] > 1.3 and
                weekly_uptrend_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below weekly S3 with volume spike and weekly downtrend
            elif (low[i] < camarilla_s3_aligned[i] and 
                  vol_spike_aligned[i] > 1.3 and
                  weekly_downtrend_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below weekly S3 (mean reversion)
            if close[i] < camarilla_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above weekly R3 (mean reversion)
            if close[i] > camarilla_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals