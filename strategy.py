#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dTrend
# Hypothesis: Use daily EMA34 for trend direction (1d) and Camarilla R1/S1 levels (1d) for breakout entries on 4h.
# Long when price breaks above R1 with daily uptrend, short when breaks below S1 with daily downtrend.
# Volume confirmation ensures breakout validity. Designed for low trade frequency (<50/year) to minimize fee drag.
# Works in bull/bear via trend filter; avoids whipsaws in sideways markets.

name = "4h_Camarilla_R1_S1_Breakout_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if df_1d.empty:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous day (OHLC)
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # Use previous day's OHLC (avoid look-ahead)
    shift_idx = len(df_1d)  # will be handled by align_htf_to_ltf
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h (wait for previous day's close)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation (20-period average on 4h)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20) + 5  # Need enough history
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or \
           np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or \
           np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: break above R1 with daily uptrend and volume
            if close[i] > camarilla_r1_aligned[i] and ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with daily downtrend and volume
            elif close[i] < camarilla_s1_aligned[i] and ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 or trend turns down
            if close[i] < camarilla_s1_aligned[i] or ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above R1 or trend turns up
            if close[i] > camarilla_r1_aligned[i] or ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals