#!/usr/bin/env python3
# 4h_Keltner_Breakout_Volume_Trend
# Hypothesis: Keltner Channel breakouts with volume confirmation and 1-day trend filter capture
# sustained momentum moves. Uses 20-period EMA with 2.0x ATR for bands. Breakouts above upper band
# signal longs when above 1d EMA50; breakdowns below lower band signal shorts when below 1d EMA50.
# Volume filter requires 1.5x 6-period average to avoid false breakouts. Designed for low trade
# frequency in both bull and bear markets by following higher timeframe trend.

name = "4h_Keltner_Breakout_Volume_Trend"
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
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Keltner Channel: 20-period EMA ± 2.0 * ATR(10)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # True Range and ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    upper_band = ema_20 + 2.0 * atr_10
    lower_band = ema_20 - 2.0 * atr_10
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (6-period average = 1.5 days for 4h timeframe)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 6)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 10, 50)  # Enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or \
           np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper band, above 1d EMA50, volume confirmation
            if close[i] > upper_band[i] and close[i] > ema_50_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band, below 1d EMA50, volume confirmation
            elif close[i] < lower_band[i] and close[i] < ema_50_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below EMA20 or below 1d EMA50
            if close[i] < ema_20[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above EMA20 or above 1d EMA50
            if close[i] > ema_20[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals