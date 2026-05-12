#!/usr/bin/env python3
# 1h_4d1d_Camarilla_R1S1_Breakout_4hTrend_Volume
# Hypothesis: Uses daily Camarilla pivot levels (R1/S1) for breakout entries on 1h timeframe.
# Trend filtered by 4h EMA50 to align with higher timeframe direction.
# Volume confirmation (>1.5x 20-period average) ensures institutional participation.
# Session filter (08-20 UTC) reduces noise trades.
# Designed for low trade frequency (<200 total 1h trades) to minimize fee drag.
# Works in bull/bear markets by following 4h trend direction while using daily Camarilla levels for precise entries.

name = "1h_4d1d_Camarilla_R1S1_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for Camarilla calculation
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close[0] = close_1d[0]  # Handle first value
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    # Camarilla formulas
    range_1d = prev_high - prev_low
    camarilla_r1 = prev_close + (range_1d * 1.1 / 12)
    camarilla_s1 = prev_close - (range_1d * 1.1 / 12)
    
    # Align daily indicators to 1h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(camarilla_r1_aligned[i]) or
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 + volume spike + price above 4h EMA50 + session
            if (close[i] > camarilla_r1_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_4h_aligned[i] and
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Camarilla S1 + volume spike + price below 4h EMA50 + session
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_4h_aligned[i] and
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters below Camarilla R1 OR closes below 4h EMA50
            if (close[i] < camarilla_r1_aligned[i]) or \
               close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price re-enters above Camarilla S1 OR closes above 4h EMA50
            if (close[i] > camarilla_s1_aligned[i]) or \
               close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals