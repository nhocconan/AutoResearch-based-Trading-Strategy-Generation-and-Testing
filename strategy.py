#!/usr/bin/env python3
# 4h_4H_1D_1W_Momentum_Trend_Breakout
# Hypothesis: Combines 4h momentum (ROC), 1d trend (EMA50), and 1w trend (EMA20) for directional bias.
# Enters on breakouts of 4h Donchian channels with volume confirmation.
# Exits on opposite Donchian break or trend reversal.
# Designed for low trade frequency (15-25/year) to minimize fee drift.
# Works in bull via trend-following breakouts and in bear via filtered momentum.

name = "4h_4H_1D_1W_Momentum_Trend_Breakout"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA20 for trend
    ema_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Daily EMA50 for trend
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 4h Donchian channel (20-period)
    lookback = 20
    def highest(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.max(arr[i-p+1:i+1])
        return res
    def lowest(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.min(arr[i-p+1:i+1])
        return res
    dc_up = highest(high, lookback)
    dc_low = lowest(low, lookback)
    
    # 4h ROC(10) for momentum
    def roc(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p, len(arr)):
                if arr[i-p] != 0:
                    res[i] = (arr[i] - arr[i-p]) / arr[i-p] * 100
        return res
    roc_10 = roc(close, 10)
    
    # Volume confirmation (24-period average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 24)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 24, 10) + 5  # need enough history
    
    for i in range(start_idx, n):
        if np.isnan(dc_up[i]) or np.isnan(dc_low[i]) or \
           np.isnan(ema_1w_aligned[i]) or np.isnan(ema_1d_aligned[i]) or \
           np.isnan(roc_10[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.8x average
        volume_confirm = volume[i] > 1.8 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: bullish momentum + price breaks above Donchian upper + uptrend on both TFs
            if roc_10[i] > 0 and close[i] > dc_up[i] and volume_confirm \
               and close[i] > ema_1d_aligned[i] and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: bearish momentum + price breaks below Donchian lower + downtrend on both TFs
            elif roc_10[i] < 0 and close[i] < dc_low[i] and volume_confirm \
                 and close[i] < ema_1d_aligned[i] and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish momentum OR price breaks below Donchian lower OR trend breaks
            if roc_10[i] < 0 or close[i] < dc_low[i] or close[i] < ema_1d_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish momentum OR price breaks above Donchian upper OR trend breaks
            if roc_10[i] > 0 or close[i] > dc_up[i] or close[i] > ema_1d_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals