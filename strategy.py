#!/usr/bin/env python3
"""
4h_Pullback_to_EMA50_1dTrend_Filter
Hypothesis: Buy dips to EMA50 in uptrend, sell rallies to EMA50 in downtrend, using 1d EMA200 for trend filter and volume confirmation.
Works in bull by buying pullbacks in uptrend; works in bear by selling rallies in downtrend.
Mean-reversion within trend reduces false breakouts and improves win rate.
Target: 25-40 trades/year (100-160 total) to minimize fee drag.
"""

name = "4h_Pullback_to_EMA50_1dTrend_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA200 for trend filter
    ema200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_1d)):
            ema200_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema200_1d[i-1]
    
    # 1d volume SMA20 for volume confirmation
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    
    # 4h EMA50 for entry
    ema50_4h = np.full(len(close), np.nan)
    if len(close) >= 50:
        ema50_4h[49] = np.mean(close[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close)):
            ema50_4h[i] = alpha * close[i] + (1 - alpha) * ema50_4h[i-1]
    
    # Align 1d indicators to 4h
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(ema50_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h-equivalent volume from 1d data
        # 1d bar = 6 x 4h bars, so scale down
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 6.0  # Average 4h-equivalent volume from 1d data
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend and price relative to 4h EMA50
        is_uptrend = close[i] > ema200_1d_aligned[i]
        is_downtrend = close[i] < ema200_1d_aligned[i]
        near_ema50 = abs(close[i] - ema50_4h[i]) / ema50_4h[i] < 0.01  # Within 1% of EMA50
        
        if position == 0:
            # Long: price near EMA50 from below in uptrend with volume
            if close[i] <= ema50_4h[i] and is_uptrend and near_ema50 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price near EMA50 from above in downtrend with volume
            elif close[i] >= ema50_4h[i] and is_downtrend and near_ema50 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price moves above EMA50 or trend turns down
            if close[i] > ema50_4h[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price moves below EMA50 or trend turns up
            if close[i] < ema50_4h[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals