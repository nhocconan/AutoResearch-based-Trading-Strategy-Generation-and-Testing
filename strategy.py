#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_WeeklyTrend
Hypothesis: Daily breakouts from Camarilla R3/S3 levels with weekly trend filter and volume confirmation.
Targets 1d timeframe to reduce trade frequency (target: 10-25 trades/year) while using proven Camarilla structure.
Only takes long when price breaks above R3 with volume spike and weekly uptrend, short when breaks below S3 with volume spike and weekly downtrend.
Weekly trend filter provides robustness across bull/bear markets, volume confirmation reduces false breakouts.
"""

name = "1d_Camarilla_R3_S3_Breakout_WeeklyTrend"
timeframe = "1d"
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
    
    # Volume spike: >2.0x 20-period average (on 1d timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate Camarilla levels from previous day
    # R3 = Close + 1.1*(High - Low)
    # S3 = Close - 1.1*(High - Low)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # First value has no previous
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low)
    
    # Weekly trend filter: 1w EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        if (np.isnan(camarilla_r3[i]) or
            np.isnan(camarilla_s3[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 + volume spike + price above weekly EMA50
            if (close[i] > camarilla_r3[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 + volume spike + price below weekly EMA50
            elif (close[i] < camarilla_s3[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters between S3 and R3 OR closes below weekly EMA50
            if (close[i] > camarilla_s3[i] and close[i] < camarilla_r3[i]) or \
               close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters between S3 and R3 OR closes above weekly EMA50
            if (close[i] > camarilla_s3[i] and close[i] < camarilla_r3[i]) or \
               close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals