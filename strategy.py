#!/usr/bin/env python3
# 4h_Camarilla_Pivot_Positions_1dTrend_VolumeFilter
# Hypothesis: On 4h timeframe, enter long when price touches Camarilla S3 with 1d uptrend and volume spike;
# enter short when price touches Camarilla R3 with 1d downtrend and volume filter.
# Exit when price reaches opposite Camarilla level (S1/R1) or trend reverses.
# Uses 1d for trend filter and Camarilla levels for mean-reversion in ranging markets.
# Targets 20-40 trades/year for low fee drag, works in both bull/bear by fading extreme intraday levels.

name = "4h_Camarilla_Pivot_Positions_1dTrend_VolumeFilter"
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
    
    # Load 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_r3 = np.zeros_like(daily_close)
    camarilla_s3 = np.zeros_like(daily_close)
    camarilla_r1 = np.zeros_like(daily_close)
    camarilla_s1 = np.zeros_like(daily_close)
    
    for i in range(len(daily_close)):
        typical = (daily_high[i] + daily_low[i] + daily_close[i]) / 3.0
        rang = daily_high[i] - daily_low[i]
        camarilla_r3[i] = daily_close[i] + rang * 1.1 / 4.0
        camarilla_s3[i] = daily_close[i] - rang * 1.1 / 4.0
        camarilla_r1[i] = daily_close[i] + rang * 1.1 / 12.0
        camarilla_s1[i] = daily_close[i] - rang * 1.1 / 12.0
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: 24-period moving average (24*4h = 4d)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Align all to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema1d_trend = ema34_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price touches S3 with 1d uptrend and volume spike
            if (low[i] <= s3_val * 1.001 and  # Allow small tolerance for touch
                close[i] > ema1d_trend and
                volume[i] > vol_ma_val * 1.8):
                signals[i] = 0.25
                position = 1
            # SHORT: Price touches R3 with 1d downtrend and volume spike
            elif (high[i] >= r3_val * 0.999 and  # Allow small tolerance for touch
                  close[i] < ema1d_trend and
                  volume[i] > vol_ma_val * 1.8):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reaches S1 (mean reversion target) or trend turns down
            if (high[i] >= s1_val * 0.999 or  # Touch or cross S1
                close[i] < ema1d_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price reaches R1 (mean reversion target) or trend turns up
            if (low[i] <= r1_val * 1.001 or  # Touch or cross R1
                close[i] > ema1d_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals