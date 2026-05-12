#!/usr/bin/env python3
"""
1h_4d_1d_Camarilla_R1_S1_Breakout_TrendVol_v1
Hypothesis: 1-hour breakouts from Camarilla R1/S1 levels (based on daily price action) with 4-hour trend filter and volume spike confirmation.
Uses 4h trend to filter direction, daily Camarilla levels for structure, and 1h for precise entry timing.
Targets 15-37 trades/year by combining tight breakout conditions with volume confirmation and trend filter.
Designed to work in both bull and bear markets via 4h trend filter and volume confirmation to avoid false breakouts.
Focuses on stronger breakout structure with daily timeframe for reliability.
"""

name = "1h_4d_1d_Camarilla_R1_S1_Breakout_TrendVol_v1"
timeframe = "1h"
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
    
    # Volume spike: >1.8x 24-period average (on 1h timeframe)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Avoid look-ahead: only use previous day's data
    range_ = prev_high - prev_low
    R1 = prev_close + 1.1 * range_ / 12
    S1 = prev_close - 1.1 * range_ / 12
    
    # Align Camarilla levels to 1h timeframe (wait for 1d bar to close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        if (np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + price above 4h EMA50
            if (close[i] > R1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 + volume spike + price below 4h EMA50
            elif (close[i] < S1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters between S1 and R1 OR closes below 4h EMA50
            if (close[i] > S1_aligned[i] and close[i] < R1_aligned[i]) or \
               close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price re-enters between S1 and R1 OR closes above 4h EMA50
            if (close[i] > S1_aligned[i] and close[i] < R1_aligned[i]) or \
               close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals