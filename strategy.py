#!/usr/bin/env python3

# 12h_1d_HighLowBreakout_VolumeTrend
# Hypothesis: Breakouts from prior 1-day high/low levels with volume confirmation and 1-day trend filter.
# Works in bull markets via breakouts above prior day high in uptrend, and in bear markets via breakdowns below prior day low in downtrend.
# Uses 12h timeframe to limit trades (target: 12-37/year) while capturing meaningful daily structure.
# Volume spike (>2x 30-period average) confirms breakout strength, reducing false signals.

name = "12h_1d_HighLowBreakout_VolumeTrend"
timeframe = "12h"
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
    
    # Volume spike: >2.0x 30-period average (on 12h timeframe)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # 1d data for prior day high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Prior day high/low (shifted by 1 to avoid look-ahead)
    prev_day_high = pd.Series(df_1d['high']).shift(1).values
    prev_day_low = pd.Series(df_1d['low']).shift(1).values
    
    # Align prior day high/low to 12h timeframe (wait for daily bar to close)
    prev_day_high_aligned = align_htf_to_ltf(prices, df_1d, prev_day_high)
    prev_day_low_aligned = align_htf_to_ltf(prices, df_1d, prev_day_low)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(prev_day_high_aligned[i]) or
            np.isnan(prev_day_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above prior day high + volume spike + price above 1d EMA50
            if (close[i] > prev_day_high_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below prior day low + volume spike + price below 1d EMA50
            elif (close[i] < prev_day_low_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters between prior day low and high OR closes below 1d EMA50
            if (close[i] > prev_day_low_aligned[i] and close[i] < prev_day_high_aligned[i]) or \
               close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters between prior day low and high OR closes above 1d EMA50
            if (close[i] > prev_day_low_aligned[i] and close[i] < prev_day_high_aligned[i]) or \
               close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals