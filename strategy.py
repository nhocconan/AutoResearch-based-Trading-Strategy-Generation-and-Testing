#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_12hEMA34_Trend_VolumeSpike
# Hypothesis: 4-hour timeframe with 12-hour trend filter using EMA34 and volume confirmation.
# Uses Camarilla R1/S1 levels from 12-hour pivot calculation for breakout/breakdown.
# Trend filter: price above/below 34 EMA on 12h timeframe.
# Volume confirmation: volume > 2.0 * 20-period average.
# Works in bull markets (breakouts continue with trend) and bear markets (mean reversion from extremes via short entries).
# Target: 75-200 total trades over 4 years = 19-50/year.

name = "4H_Camarilla_R1_S1_Breakout_12hEMA34_Trend_VolumeSpike"
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
    
    # 12h EMA34 for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 12h data for Camarilla R1/S1 levels
    prev_close_12h = df_12h['close'].shift(1).values
    prev_high_12h = df_12h['high'].shift(1).values
    prev_low_12h = df_12h['low'].shift(1).values
    rang_12h = prev_high_12h - prev_low_12h
    R1_12h = prev_close_12h + 1.1 * rang_12h * 1.1 / 4
    S1_12h = prev_close_12h - 1.1 * rang_12h * 1.1 / 4
    
    # Align 12h Camarilla levels to 4h timeframe
    R1_12h_aligned = align_htf_to_ltf(prices, df_12h, R1_12h)
    S1_12h_aligned = align_htf_to_ltf(prices, df_12h, S1_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup for EMA34
        if (np.isnan(R1_12h_aligned[i]) or 
            np.isnan(S1_12h_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R1_12h + volume spike + price above 12h EMA34 (uptrend)
            if (close[i] > R1_12h_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1_12h + volume spike + price below 12h EMA34 (downtrend)
            elif (close[i] < S1_12h_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_34_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters previous 12h H-L range (between S1 and R1) OR closes below 12h EMA34
            if (close[i] < R1_12h_aligned[i] and close[i] > S1_12h_aligned[i]) or \
               close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters previous 12h H-L range (between S1 and R1) OR closes above 12h EMA34
            if (close[i] < R1_12h_aligned[i] and close[i] > S1_12h_aligned[i]) or \
               close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals