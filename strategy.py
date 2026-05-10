#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
# Hypothesis: Camarilla pivot levels (R1, S1) act as strong support/resistance. 
# A breakout above R1 with volume and 12h trend confirmation indicates bullish momentum,
# while a breakdown below S1 with volume and 12h trend confirmation indicates bearish momentum.
# Works in both bull and bear markets by following the trend defined by 12h EMA50.
# Uses volume confirmation to avoid false breakouts. Designed for low trade frequency (20-40/year).

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend (smooth, lag-appropriate)
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Get daily data for Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # R1 = close + (high - low) * 1.12
    # S1 = close - (high - low) * 1.12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.12
    
    # Use previous day's levels (shift by 1)
    prev_r1 = camarilla_r1.shift(1).values
    prev_s1 = camarilla_s1.shift(1).values
    
    # Align daily levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, prev_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, prev_s1)
    
    # Volume confirmation (20-period average on 4h = ~10 days)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20) + 5  # need enough history for calculations
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above R1 with volume, above 12h EMA50 (uptrend)
            if close[i] > r1_aligned[i] and volume_confirm and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, below 12h EMA50 (downtrend)
            elif close[i] < s1_aligned[i] and volume_confirm and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S1 or breaks below 12h EMA50
            if close[i] < s1_aligned[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R1 or breaks above 12h EMA50
            if close[i] > r1_aligned[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals