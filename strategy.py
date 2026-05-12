#!/usr/bin/env python3
"""
6h_1D_1W_Camarilla_R3_S3_Breakout_TrendVol_v2
Hypothesis: Camarilla R3/S3 breakout with weekly trend filter and volume confirmation.
Targets 6h timeframe for balanced trade frequency (target: 20-40 trades/year).
Uses daily Camarilla levels for precise entry/exit, weekly trend for direction filter,
and volume spike to confirm institutional interest. Works in bull/bear via trend filter.
"""

name = "6h_1D_1W_Camarilla_R3_S3_Breakout_TrendVol_v2"
timeframe = "6h"
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
    
    # Volume spike: >2.0x 30-period average (on 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    daily_range = daily_high - daily_low
    
    # Camarilla R3, S3, R4, S4 levels
    camarilla_r3 = daily_close + 1.1 * daily_range * 1.1 / 2
    camarilla_s3 = daily_close - 1.1 * daily_range * 1.1 / 2
    camarilla_r4 = daily_close + 1.1 * daily_range * 1.5 / 2
    camarilla_s4 = daily_close - 1.1 * daily_range * 1.5 / 2
    
    # Align Camarilla levels to 6h timeframe (wait for daily bar to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA21 for trend filter
    ema_21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above R3 with volume spike and weekly uptrend
            if (close[i] > camarilla_r3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_21_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Break below S3 with volume spike and weekly downtrend
            elif (close[i] < camarilla_s3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_21_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Break below S4 or weekly trend turns down
            if (close[i] < camarilla_s4_aligned[i] or 
                close[i] < ema_21_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Break above R4 or weekly trend turns up
            if (close[i] > camarilla_r4_aligned[i] or 
                close[i] > ema_21_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals