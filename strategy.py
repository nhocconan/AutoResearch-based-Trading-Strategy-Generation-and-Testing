#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1wTrend_Volume
# Hypothesis: Camarilla pivot levels (R3/S3) from daily data act as strong support/resistance.
# In uptrend (price > weekly EMA34), buy at S3 with volume confirmation.
# In downtrend (price < weekly EMA34), sell at R3 with volume confirmation.
# Uses 1-day Camarilla and 1-week EMA34 for multi-timeframe alignment.
# Designed for low trade frequency (20-40/year) to minimize fee drag.

name = "4h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
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
    
    # Get 1-day data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels (R3, S3) for each day
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    camarilla_range = high_1d - low_1d
    r3_1d = close_1d + 1.1 * camarilla_range
    s3_1d = close_1d - 1.1 * camarilla_range
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation (24-period average on 4h = 4 days)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 24) + 5  # need enough history
    
    for i in range(start_idx, n):
        if np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or \
           np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price at S3 support in uptrend (price > weekly EMA34) with volume
            if close[i] <= s3_1d_aligned[i] * 1.005 and \
               close[i] > ema_34_1w_aligned[i] and \
               volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price at R3 resistance in downtrend (price < weekly EMA34) with volume
            elif close[i] >= r3_1d_aligned[i] * 0.995 and \
                 close[i] < ema_34_1w_aligned[i] and \
                 volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses weekly EMA34 down OR reaches R3
            if close[i] < ema_34_1w_aligned[i] or close[i] >= r3_1d_aligned[i] * 0.995:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses weekly EMA34 up OR reaches S3
            if close[i] > ema_34_1w_aligned[i] or close[i] <= s3_1d_aligned[i] * 1.005:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals