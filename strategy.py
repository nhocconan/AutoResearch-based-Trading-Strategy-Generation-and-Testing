#!/usr/bin/env python3
"""
1d_1w_Camarilla_R3_S3_Breakout_TrendVol_v1
Hypothesis: Daily breakouts from weekly-based Camarilla R3/S3 levels with weekly trend filter and volume confirmation.
Designed for 1d timeframe to capture multi-day moves with low trade frequency (target: 7-25 trades/year).
Uses weekly timeframe for trend filter and daily for entry signals to work in both bull and bear markets.
Only takes long when price breaks above weekly R3 with volume spike and weekly uptrend,
short when breaks below weekly S3 with volume spike and weekly downtrend.
"""

name = "1d_1w_Camarilla_R3_S3_Breakout_TrendVol_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >2.0x 20-period average (on 1d timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly data for Camarilla levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous weekly bar
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Avoid look-ahead: only use previous week's data
    range_ = prev_high - prev_low
    R3 = prev_close + 1.1 * range_ / 4
    S3 = prev_close - 1.1 * range_ / 4
    
    # Align weekly Camarilla levels to daily timeframe (wait for weekly bar to close)
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(R3_aligned[i]) or
            np.isnan(S3_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above weekly R3 + volume spike + price above weekly EMA34
            if (close[i] > R3_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below weekly S3 + volume spike + price below weekly EMA34
            elif (close[i] < S3_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters between S3 and R3 OR closes below weekly EMA34
            if (close[i] > S3_aligned[i] and close[i] < R3_aligned[i]) or \
               close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters between S3 and R3 OR closes above weekly EMA34
            if (close[i] > S3_aligned[i] and close[i] < R3_aligned[i]) or \
               close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals