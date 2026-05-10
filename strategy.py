#!/usr/bin/env python3
# 4h_VWAP_Trend_Momentum_1d
# Hypothesis: VWAP acts as a dynamic support/resistance level on 4h, with price bouncing off VWAP in trending markets. 
# Uses 1d EMA50 for trend filter and volume spike for momentum confirmation. Designed for 4h to achieve 20-40 trades/year,
# working in both bull (buy dips to VWAP in uptrend) and bear (sell rallies to VWAP in downtrend) markets.

name = "4h_VWAP_Trend_Momentum_1d"
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
    
    # 1d data for trend filter and volume context
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # VWAP calculation for 4h
    typical_price = (high + low + close) / 3.0
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike detection: current volume > 2.0 * 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20_1d = mean_arr(volume_1d, 20)
    
    # Align indicators to 4h timeframe (wait for 1d bar to close)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for VWAP and indicators
    
    for i in range(start_idx, n):
        if np.isnan(vwap[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price crosses above VWAP, above EMA50, with volume spike
            if close[i] > vwap[i] and close[i] > ema_50_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below VWAP, below EMA50, with volume spike
            elif close[i] < vwap[i] and close[i] < ema_50_aligned[i] and volume[i] > 2.0 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below VWAP or below EMA50
            if close[i] < vwap[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above VWAP or above EMA50
            if close[i] > vwap[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals