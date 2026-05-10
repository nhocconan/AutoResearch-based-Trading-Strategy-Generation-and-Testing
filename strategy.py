#!/usr/bin/env python3
"""
4h_SMA50_Breakout_Volume_Confirm
Hypothesis: Price breaks above SMA50 for longs or below SMA50 for shorts on 4h timeframe, confirmed by volume spike (>1.5x 10-period SMA) and filtered by 1d EMA200 trend. This captures momentum in trending markets while avoiding counter-trend trades. The SMA50 acts as dynamic support/resistance, and volume confirms institutional participation. Works in bull/bear by aligning with higher timeframe trend.
Target: 30-50 trades/year (120-200 total) to minimize fee drag.
"""

name = "4h_SMA50_Breakout_Volume_Confirm"
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
    
    # 4h SMA50 for dynamic support/resistance
    sma50 = np.full(n, np.nan)
    if n >= 50:
        sma50[49] = np.mean(close[:50])
        for i in range(50, n):
            sma50[i] = (sma50[i-1] * 49 + close[i]) / 50
    
    # 4h volume SMA10 for volume confirmation
    vol_sma10 = np.full(n, np.nan)
    if n >= 10:
        vol_sma10[9] = np.mean(volume[:10])
        for i in range(10, n):
            vol_sma10[i] = (vol_sma10[i-1] * 9 + volume[i]) / 10
    
    # 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA200 for trend filter
    ema200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_1d)):
            ema200_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema200_1d[i-1]
    
    # Align 1d EMA200 to 4h
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 200)  # Wait for SMA50 and EMA200
    
    for i in range(start_idx, n):
        if np.isnan(sma50[i]) or np.isnan(vol_sma10[i]) or np.isnan(ema200_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 10-period SMA
        volume_confirm = volume[i] > 1.5 * vol_sma10[i]
        
        # Trend and price relative to SMA50
        is_uptrend = close[i] > ema200_1d_aligned[i]
        is_downtrend = close[i] < ema200_1d_aligned[i]
        price_above_sma50 = close[i] > sma50[i]
        price_below_sma50 = close[i] < sma50[i]
        
        if position == 0:
            # Long: price breaks above SMA50, in uptrend, with volume
            if price_above_sma50 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below SMA50, in downtrend, with volume
            elif price_below_sma50 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below SMA50 or trend turns down
            if not price_above_sma50 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above SMA50 or trend turns up
            if not price_below_sma50 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals