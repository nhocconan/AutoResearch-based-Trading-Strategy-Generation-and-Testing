#!/usr/bin/env python3
# 6h_Keltner_Channel_Breakout_12hTrend_Volume
# Hypothesis: 6-hour breakouts from Keltner Channel (ATR-based) with 12h trend filter and volume confirmation.
# Uses 12h EMA50 for trend direction and 12h ATR(10) for channel width. Volume requires 1.5x 20-period average.
# Designed for 6h to achieve 12-37 trades/year. Works in both bull and bear markets by following higher timeframe trend.

name = "6h_Keltner_Channel_Breakout_12hTrend_Volume"
timeframe = "6h"
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
    
    # 12h data for trend filter and Keltner Channel
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 12h ATR(10) for Keltner Channel width
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR
    atr_10_12h = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel: Upper = EMA + 2*ATR, Lower = EMA - 2*ATR
    keltner_upper = ema_50_12h + 2.0 * atr_10_12h
    keltner_lower = ema_50_12h - 2.0 * atr_10_12h
    
    # Volume confirmation: 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p - 1, len(arr)):
                res[i] = np.mean(arr[i - p + 1:i + 1])
        return res
    vol_ma_20 = mean_arr(volume, 20)
    
    # Align 12h indicators to 6h timeframe (wait for 12h bar to close)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    keltner_upper_aligned = align_htf_to_ltf(prices, df_12h, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_12h, keltner_lower)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(keltner_upper_aligned[i]) or \
           np.isnan(keltner_lower_aligned[i]) or np.isnan(vol_ma_20_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Keltner Upper, above 12h EMA50, strong volume
            if close[i] > keltner_upper_aligned[i] and close[i] > ema_50_12h_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Keltner Lower, below 12h EMA50, strong volume
            elif close[i] < keltner_lower_aligned[i] and close[i] < ema_50_12h_aligned[i] and volume[i] > 1.5 * vol_ma_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below Keltner Lower or below 12h EMA50
            if close[i] < keltner_lower_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above Keltner Upper or above 12h EMA50
            if close[i] > keltner_upper_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals