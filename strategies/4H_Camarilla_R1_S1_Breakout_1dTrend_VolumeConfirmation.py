#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation
# Hypothesis: 4-hour timeframe with daily trend filter and volume confirmation.
# Uses Camarilla R1/S1 levels from daily pivot calculation for breakout/breakdown.
# Trend filter: price above/below 100 EMA on 1d timeframe.
# Volume confirmation: volume > 2.0 * 20-period average.
# Works in bull markets (breakouts continue with trend) and bear markets (mean reversion from extremes via short entries).
# Target: 50-150 total trades over 4 years = 12-37/year.

name = "4H_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirmation"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA100 for trend filter
    ema_100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    # 1d data for Camarilla R1/S1 levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    rang = prev_high - prev_low
    R1 = prev_close + 1.1 * rang * 1.1 / 4
    S1 = prev_close - 1.1 * rang * 1.1 / 4
    
    # Align daily Camarilla levels to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMA100
        if (np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or 
            np.isnan(ema_100_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1 + volume spike + price above 1d EMA100 (uptrend)
            if (close[i] > R1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_100_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + volume spike + price below 1d EMA100 (downtrend)
            elif (close[i] < S1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_100_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous day's H-L range (between S1 and R1) OR closes below 1d EMA100
            if (close[i] < R1_aligned[i] and close[i] > S1_aligned[i]) or \
               close[i] < ema_100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters previous day's H-L range (between S1 and R1) OR closes above 1d EMA100
            if (close[i] < R1_aligned[i] and close[i] > S1_aligned[i]) or \
               close[i] > ema_100_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals