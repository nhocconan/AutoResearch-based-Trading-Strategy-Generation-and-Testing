#!/usr/bin/env python3
"""
6h_Keltner_Channel_Volume_Breakout
Hypothesis: Uses Keltner Channel (20, ATR10) on 6h timeframe with volume confirmation
to capture breakouts in both bull and bear markets. Filters trades using 12h EMA50
trend direction to avoid counter-trend entries. Designed for low trade frequency
(15-25/year) to minimize fee drag while capturing strong momentum moves.
"""

name = "6h_Keltner_Channel_Volume_Breakout"
timeframe = "6h"
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
    
    # Calculate ATR for Keltner Channel
    tr1 = np.maximum(high[1:], close[:-1]) - np.minimum(low[1:], close[:-1])
    tr1 = np.insert(tr1, 0, high[0] - low[0])
    
    atr_period = 10
    def sum_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.sum(arr[i-p+1:i+1])
        return res
    
    tr_sum = sum_arr(tr1, atr_period)
    atr = tr_sum / atr_period
    
    # Keltner Channel (20, ATR10)
    kc_period = 20
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    
    ma = mean_arr(close, kc_period)
    upper = ma + 2.0 * atr
    lower = ma - 2.0 * atr
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation
    vol_ma_period = 20
    vol_ma = mean_arr(volume, vol_ma_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(kc_period, atr_period, vol_ma_period) + 5
    
    for i in range(start_idx, n):
        if np.isnan(ma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or \
           np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above upper Keltner band with volume, above 12h EMA50
            if close[i] > upper[i] and volume_confirm and close[i] > ema_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Keltner band with volume, below 12h EMA50
            elif close[i] < lower[i] and volume_confirm and close[i] < ema_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below middle line or breaks below 12h EMA50
            if close[i] < ma[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above middle line or breaks above 12h EMA50
            if close[i] > ma[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals