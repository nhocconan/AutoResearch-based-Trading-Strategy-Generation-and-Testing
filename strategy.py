#!/usr/bin/env python3
# 4h_KAMA_Trend_Follow_1dTrend_Volume
# Hypothesis: 4-hour trend following using Kaufman's Adaptive Moving Average (KAMA) with daily trend filter (EMA34) and volume confirmation.
# KAMA adapts to market noise - faster in trends, slower in ranges, reducing whipsaws. Daily EMA34 ensures trades align with higher timeframe trend.
# Volume confirmation filters weak breakouts. Designed for 4h to achieve 20-50 trades/year, suitable for both bull and bear markets.

name = "4h_KAMA_Trend_Follow_1dTrend_Volume"
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
    
    # Daily data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
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
    
    # KAMA (4-period ER, 2 and 30 for fast/slow SC)
    def kama(close, er_period=4, fast_sc=2, slow_sc=30):
        n = len(close)
        kama_out = np.full(n, np.nan)
        if n < er_period:
            return kama_out
        change = np.abs(close[er_period:] - close[:-er_period])
        abs_diff = np.abs(np.diff(close))
        er = np.zeros(n)
        er[:er_period] = np.nan
        for i in range(er_period, n):
            if abs_diff[i-er_period+1:i+1].sum() > 0:
                er[i] = change[i-er_period] / abs_diff[i-er_period+1:i+1].sum()
            else:
                er[i] = 0
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama_out[er_period-1] = close[er_period-1]
        for i in range(er_period, n):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_val = kama(close)
    
    # Align daily indicators to 4h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i]) or np.isnan(kama_val[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, above daily EMA34, strong volume
            if close[i] > kama_val[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, below daily EMA34, strong volume
            elif close[i] < kama_val[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below KAMA or below daily EMA34
            if close[i] < kama_val[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above KAMA or above daily EMA34
            if close[i] > kama_val[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals