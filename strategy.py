#!/usr/bin/env python3
# 1h_4h_1D_Camarilla_R1_S1_Breakout_TrendVolume
# Hypothesis: 1-hour entries aligned with 4-hour trend and daily Camarilla levels for breakout confirmation.
# Uses 4h EMA50 for trend filter, daily R1/S1 for breakout levels, and volume spike for confirmation.
# Targets 15-37 trades/year by requiring confluence of 4h trend, daily level break, and volume spike.
# Works in bull markets via breakout continuation and in bear markets via mean-reversion from extremes.

name = "1h_4h_1D_Camarilla_R1_S1_Breakout_TrendVolume"
timeframe = "1h"
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
    
    # Volume spike: >2.0x 24-period average (6 hours of 1h bars)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily Camarilla R1 and S1 from previous day
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    rang_1d = prev_high_1d - prev_low_1d
    R1_1d = prev_close_1d + 1.1 * rang_1d * 1.0 / 4
    S1_1d = prev_close_1d - 1.1 * rang_1d * 1.0 / 4
    
    # Align daily levels to 1h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(R1_1d_aligned[i]) or 
            np.isnan(S1_1d_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + price above 4h EMA50 (uptrend)
            if (close[i] > R1_1d_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + volume spike + price below 4h EMA50 (downtrend)
            elif (close[i] < S1_1d_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous day's H-L range OR closes below 4h EMA50
            if (close[i] < R1_1d_aligned[i] and close[i] > S1_1d_aligned[i]) or \
               close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price re-enters previous day's H-L range OR closes above 4h EMA50
            if (close[i] < R1_1d_aligned[i] and close[i] > S1_1d_aligned[i]) or \
               close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals