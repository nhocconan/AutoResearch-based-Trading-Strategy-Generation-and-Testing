#!/usr/bin/env python3
"""
1d_1w_Camarilla_R3_S3_Breakout_TrendVol_v1
Hypothesis: Daily breakouts from weekly Camarilla R3/S3 levels with monthly trend filter and volume spike confirmation.
Targets 1d timeframe to minimize trade frequency while using proven weekly Camarilla levels and monthly trend filter.
Only takes long when price breaks above weekly R3 with volume spike and monthly uptrend, short when breaks below weekly S3 with volume spike and monthly downtrend.
Designed to work in both bull and bear markets via trend filter and volume confirmation to avoid false breakouts.
Focuses on stronger breakout levels (R3/S3) for higher quality signals.
"""

name = "1d_1w_Camarilla_R3_S3_Breakout_TrendVol_v1"
timeframe = "1d"
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
    volume = prices['volume'].values
    
    # Volume spike: >2.0x 20-period average (on daily timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous weekly bar
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Avoid look-ahead: only use previous week's data
    range_ = prev_high - prev_low
    R3 = prev_close + 1.1 * range_ / 4
    S3 = prev_close - 1.1 * range_ / 4
    
    # Align Camarilla levels to daily timeframe (wait for weekly bar to close)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Monthly data for trend filter
    df_1M = get_htf_data(prices, '1M')
    if len(df_1M) < 2:
        return np.zeros(n)
    
    # Monthly EMA34 for trend filter
    ema_34_1M = pd.Series(df_1M['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1M_aligned = align_htf_to_ltf(prices, df_1M, ema_34_1M)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(ema_34_1M_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly R3 + volume spike + price above monthly EMA34
            if (close[i] > R3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1M_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S3 + volume spike + price below monthly EMA34
            elif (close[i] < S3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1M_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters between S3 and R3 OR closes below monthly EMA34
            if (close[i] > S3_aligned[i] and close[i] < R3_aligned[i]) or \
               close[i] < ema_34_1M_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters between S3 and R3 OR closes above monthly EMA34
            if (close[i] > S3_aligned[i] and close[i] < R3_aligned[i]) or \
               close[i] > ema_34_1M_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals