#!/usr/bin/env python3
# 4h_12h_Camarilla_R3S3_Breakout_VolumeSpike
# Hypothesis: Breakouts from 12h Camarilla R3/S3 levels with volume spike confirmation and 12h EMA50 trend filter.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion from extremes (short at S3, long at R3).
# Uses 12h for structure/trend and 4h for execution to limit trade frequency (target: 20-50 trades/year).

name = "4h_12h_Camarilla_R3S3_Breakout_VolumeSpike"
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
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 12h data for trend filter and Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 12h data for Camarilla R3/S3 levels
    prev_close_12h = df_12h['close'].shift(1).values
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    rang_12h = prev_high_12h - prev_low_12h
    R3_12h = prev_close_12h + 1.1 * rang_12h * 1.1 / 2
    S3_12h = prev_close_12h - 1.1 * rang_12h * 1.1 / 2
    
    # Align 12h Camarilla levels to 4h timeframe
    R3_12h_aligned = align_htf_to_ltf(prices, df_12h, R3_12h)
    S3_12h_aligned = align_htf_to_ltf(prices, df_12h, S3_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMA50
        if (np.isnan(R3_12h_aligned[i]) or 
            np.isnan(S3_12h_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3_12h + volume spike + price above 12h EMA50 (uptrend)
            if (close[i] > R3_12h_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3_12h + volume spike + price below 12h EMA50 (downtrend)
            elif (close[i] < S3_12h_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous 12h H-L range (between S3 and R3) OR closes below 12h EMA50
            if (close[i] < R3_12h_aligned[i] and close[i] > S3_12h_aligned[i]) or \
               close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters previous 12h H-L range (between S3 and R3) OR closes above 12h EMA50
            if (close[i] < R3_12h_aligned[i] and close[i] > S3_12h_aligned[i]) or \
               close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals