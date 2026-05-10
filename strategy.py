#!/usr/bin/env python3
# 1d_KAMA_Trend_Follow_1wTrend_Volume
# Hypothesis: Daily KAMA trend with weekly trend filter and volume confirmation.
# KAMA adapts to market noise, reducing whipsaws in sideways markets.
# Weekly trend filter ensures we only trade in the direction of the higher timeframe trend.
# Volume confirmation ensures breakout strength.
# Designed for 1d to achieve 7-25 trades/year, suitable for both bull and bear markets.

name = "1d_KAMA_Trend_Follow_1wTrend_Volume"
timeframe = "1d"
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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Daily KAMA (adaptive moving average)
    def kama(close, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)
        er = np.zeros_like(close)
        for i in range(len(close)):
            if volatility[i] > 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    # Align weekly indicators to daily timeframe (wait for weekly bar to close)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Align daily KAMA and volume MA
    kama_vals = kama(close)
    vol_ma_20_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(kama_vals[i]) or np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, above weekly EMA34, strong volume
            if close[i] > kama_vals[i] and close[i] > ema_34_1w_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, below weekly EMA34, strong volume
            elif close[i] < kama_vals[i] and close[i] < ema_34_1w_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below KAMA or below weekly EMA34
            if close[i] < kama_vals[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above KAMA or above weekly EMA34
            if close[i] > kama_vals[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals