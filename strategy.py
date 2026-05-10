#!/usr/bin/env python3
"""
4h_MidPoint_Rebound_1dTrend
Hypothesis: Price rebounds from daily midpoint (average of daily high/low) with 1d trend filter and volume confirmation.
Long when price > daily midpoint and above 1d EMA50 with volume surge; short when price < daily midpoint and below 1d EMA50 with volume surge.
Works in bull/bear by capturing mean-reversion within the dominant daily trend.
Target: 20-40 trades/year (80-160 total) to minimize fee drag.
"""

name = "4h_MidPoint_Rebound_1dTrend"
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
    
    # 1d data
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Daily midpoint: (daily high + daily low) / 2
    midpoint_1d = (high_1d + low_1d) / 2
    
    # 1d EMA50 for trend filter
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    # 1d volume SMA20 for volume confirmation
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    
    # Align 1d indicators to 4h
    midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(midpoint_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 1d volume (scaled)
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 6.0  # 6x 4h bars in 1d
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend and position relative to midpoint
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        price_above_mid = close[i] > midpoint_aligned[i]
        price_below_mid = close[i] < midpoint_aligned[i]
        
        if position == 0:
            # Long: price above midpoint, in uptrend, with volume
            if price_above_mid and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price below midpoint, in downtrend, with volume
            elif price_below_mid and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls below midpoint or trend turns down
            if not price_above_mid or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises above midpoint or trend turns up
            if not price_below_mid or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals