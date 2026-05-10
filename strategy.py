#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Camarilla pivot levels from 1d timeframe provide institutional support/resistance.
# Breakouts at R3/S3 levels with daily trend confirmation and volume spikes capture
# momentum moves while minimizing false breakouts in choppy markets.
# Designed for 6h timeframe to achieve 50-150 trades over 4 years (~12-37/year) with low fee drag.
# Works in both bull and bear markets by using 1d trend filter to align with higher timeframe momentum.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once before loop (CRITICAL: avoids 45K file reads)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from 1d data
    # Formula: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # But we use standard Camarilla: based on previous day's range
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Calculate pivot range
    prev_range = phigh - plow
    
    # Camarilla levels (using previous day's close as pivot base)
    # R3 = pclose + 1.1 * prev_range
    # S3 = pclose - 1.1 * prev_range
    r3 = pclose + 1.1 * prev_range
    s3 = pclose - 1.1 * prev_range
    r4 = pclose + 1.5 * prev_range
    s4 = pclose - 1.5 * prev_range
    
    # Align Camarilla levels to 6h timeframe (wait for 1d bar to close)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d trend filter: EMA34 (standard for Camarilla strategies)
    ema_34_1d = pd.Series(pclose).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: 24-period average on 6h (equivalent to 4d on 6h chart)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24) + 5  # Need enough history for EMA and volume
    
    for i in range(start_idx, n):
        if np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or \
           np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 2.0 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long breakout: price closes above R3 with 1d uptrend and volume
            if close[i] > r3_6h[i] and close[i] > ema_34_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below S3 with 1d downtrend and volume
            elif close[i] < s3_6h[i] and close[i] < ema_34_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below S3 (reversion to mean) or trend breaks
            if close[i] < s3_6h[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above R3 (reversion to mean) or trend breaks
            if close[i] > r3_6h[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals