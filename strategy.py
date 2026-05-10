#!/usr/bin/env python3
# 12h_KAMA_Trend_1dEMA34_VolumeFilter
# Hypothesis: 12-hour trend-following using Kaufman Adaptive Moving Average (KAMA) with daily EMA34 trend filter and volume confirmation. KAMA adapts to market noise, reducing whipsaws in ranging markets while capturing trends. Daily EMA34 ensures alignment with higher timeframe trend, and volume filter confirms momentum. Designed for 12h to achieve 12-35 trades/year, suitable for both bull and bear markets by avoiding counter-trend trades and false breakouts.

name = "12h_KAMA_Trend_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # KAMA (ER=10, FAST=2, SLOW=30) - adaptive moving average
    def kama(close, er_length=10, fast=2, slow=30):
        n = len(close)
        if n < er_length:
            return np.full(n, np.nan)
        change = np.abs(np.subtract(close[er_length:], close[:-er_length]))
        volatility = np.sum(np.abs(np.diff(close[:n-er_length+1])), axis=0) if n >= er_length else 0
        # Avoid division by zero
        er = np.where(volatility != 0, change / volatility, 0)
        sc = np.power(er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1), 2)
        kama_vals = np.full(n, np.nan)
        kama_vals[er_length-1] = close[er_length-1]
        for i in range(er_length, n):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Calculate KAMA on 12h data
    kama_vals = kama(close)
    
    # Align daily indicators to 12h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_vals[i]) or np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, above daily EMA34, strong volume
            if close[i] > kama_vals[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, below daily EMA34, strong volume
            elif close[i] < kama_vals[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below KAMA or below daily EMA34
            if close[i] < kama_vals[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above KAMA or above daily EMA34
            if close[i] > kama_vals[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals