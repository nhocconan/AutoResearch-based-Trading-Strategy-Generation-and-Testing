#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend_With_Volume_Filter
# Hypothesis: Camarilla pivot levels (R1/S1) from daily act as strong support/resistance.
# Breakouts above R1 or below S1 with daily trend filter (price > 1d EMA34) and volume confirmation
# capture institutional breakouts. Works in bull (breakouts up) and bear (breakdowns down) by following
# daily trend. Low trade frequency expected due to strict breakout conditions + volume + trend filter.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_With_Volume_Filter"
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
    
    # 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's close for Camarilla calculation
    prev_close_1d = np.append([np.nan], close_1d[:-1])
    
    # Calculate Camarilla levels for each day
    range_1d = high_1d - low_1d
    R1 = prev_close_1d + (range_1d * 1.0833)
    S1 = prev_close_1d - (range_1d * 1.0833)
    
    # Daily EMA34 for trend filter
    close_ser = pd.Series(close_1d)
    ema_34_1d = close_ser.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (24-period average = 4 days on 4h chart)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34 + 24  # Need EMA34 and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or \
           np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above R1, above daily EMA34 (uptrend), volume confirmation
            if close[i] > R1_aligned[i] and close[i] > ema_34_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below daily EMA34 (downtrend), volume confirmation
            elif close[i] < S1_aligned[i] and close[i] < ema_34_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops back below R1 OR below daily EMA34
            if close[i] < R1_aligned[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises back above S1 OR above daily EMA34
            if close[i] > S1_aligned[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals