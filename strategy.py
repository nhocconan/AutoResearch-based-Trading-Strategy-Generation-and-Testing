#!/usr/bin/env python3
# 4h_Keltner_Breakout_1dTrend_Volume
# Hypothesis: 4-hour breakouts from Keltner Channel (20,2.0) with daily EMA50 trend filter and volume confirmation.
# Keltner breakouts capture volatility expansion; daily EMA50 ensures trend alignment; volume confirms strength.
# Designed for 4h to achieve 20-50 trades/year, suitable for both bull and bear markets.

name = "4h_Keltner_Breakout_1dTrend_Volume"
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
    
    # Daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Keltner Channel: 20-period ATR, 2.0 multiplier
    def true_range(h, l, c_prev):
        tr1 = h - l
        tr2 = np.abs(h - c_prev)
        tr3 = np.abs(l - c_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    tr = np.zeros_like(high)
    tr[0] = high[0] - low[0]
    for i in range(1, len(high)):
        tr[i] = true_range(high[i], low[i], close[i-1])
    
    atr = np.zeros_like(high)
    for i in range(len(high)):
        if i < 19:
            atr[i] = np.nan
        else:
            atr[i] = np.mean(tr[i-19:i+1])
    
    upper = np.zeros_like(high)
    lower = np.zeros_like(high)
    ma = np.zeros_like(high)
    for i in range(len(high)):
        if i < 19:
            upper[i] = np.nan
            lower[i] = np.nan
            ma[i] = np.nan
        else:
            ma[i] = np.mean(high[i-19:i+1])  # Typical price approximation
            upper[i] = ma[i] + 2.0 * atr[i]
            lower[i] = ma[i] - 2.0 * atr[i]
    
    # Daily volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume_1d, 20)
    
    # Align daily indicators to 4h timeframe (wait for 1d bar to close)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or \
           np.isnan(upper[i]) or np.isnan(lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Keltner, above daily EMA50, strong volume
            if close[i] > upper[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner, below daily EMA50, strong volume
            elif close[i] < lower[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below lower Keltner or below daily EMA50
            if close[i] < lower[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above upper Keltner or above daily EMA50
            if close[i] > upper[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals