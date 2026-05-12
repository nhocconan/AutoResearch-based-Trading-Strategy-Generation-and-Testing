#!/usr/bin/env python3
# 4h_1D_Camarilla_R3_S3_Breakout_Pullback
# Hypothesis: 4-hour pullbacks to daily R3/S3 levels with daily EMA34 trend filter and volume confirmation.
# In uptrends, buy pullbacks to daily R3; in downtrends, sell pullbacks to daily S3.
# Uses higher-probability entries (trend + pullback + volume) to target 25-40 trades per year.
# Works in bull markets via trend continuation and in bear markets via counter-trend bounces at extreme daily levels.

name = "4h_1D_Camarilla_R3_S3_Breakout_Pullback"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for Camarilla levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Daily Camarilla R3 and S3 from previous day
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    rang_1d = prev_high_1d - prev_low_1d
    R3_1d = prev_close_1d + 1.1 * rang_1d * 3.0 / 4
    S3_1d = prev_close_1d - 1.1 * rang_1d * 3.0 / 4
    
    # Align daily levels to 4h timeframe
    R3_1d_aligned = align_htf_to_ltf(prices, df_1d, R3_1d)
    S3_1d_aligned = align_htf_to_ltf(prices, df_1d, S3_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(R3_1d_aligned[i]) or 
            np.isnan(S3_1d_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Pullback to R3 in uptrend (price <= R3 but > S3) with volume spike
            if (close[i] <= R3_1d_aligned[i] and 
                close[i] > S3_1d_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Pullback to S3 in downtrend (price >= S3 but < R3) with volume spike
            elif (close[i] >= S3_1d_aligned[i] and 
                  close[i] < R3_1d_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S3 or closes below EMA34
            if close[i] < S3_1d_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R3 or closes above EMA34
            if close[i] > R3_1d_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals