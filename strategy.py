#!/usr/bin/env python3
"""
6h_MultiTimeframe_ElderRay_Strategy
Hypothesis: Use Elder Ray (Bull/Bear Power) from 1d with 60-day EMA trend filter to capture sustained momentum.
In bull markets: Buy when Bear Power turns positive (bullish momentum) while price above EMA60.
In bear markets: Sell when Bull Power turns negative (bearish momentum) while price below EMA60.
Adds volume confirmation to filter weak signals. Works in both regimes by adapting to Elder Ray's polarity.
Target: 15-30 trades/year (60-120 total) to minimize fee drag.
"""

name = "6h_MultiTimeframe_ElderRay_Strategy"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Elder Ray and trend
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # EMA60 for trend filter (1d)
    ema60_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 60:
        ema60_1d[59] = np.mean(close_1d[:60])
        alpha = 2 / (60 + 1)
        for i in range(60, len(close_1d)):
            ema60_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema60_1d[i-1]
    
    # Bull Power = High - EMA60
    bull_power_1d = np.full(len(high_1d), np.nan)
    for i in range(len(high_1d)):
        if not np.isnan(ema60_1d[i]):
            bull_power_1d[i] = high_1d[i] - ema60_1d[i]
    
    # Bear Power = Low - EMA60
    bear_power_1d = np.full(len(low_1d), np.nan)
    for i in range(len(low_1d)):
        if not np.isnan(ema60_1d[i]):
            bear_power_1d[i] = low_1d[i] - ema60_1d[i]
    
    # 1d volume SMA20 for confirmation
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    
    # Align 1d indicators to 6h
    ema60_1d_aligned = align_htf_to_ltf(prices, df_1d, ema60_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for EMA60
    
    for i in range(start_idx, n):
        if np.isnan(ema60_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 1d volume (scaled)
        # 1d = 4 x 6h bars, so scale down
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 4.0
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend and Elder Ray signals
        price_above_ema60 = close[i] > ema60_1d_aligned[i]
        price_below_ema60 = close[i] < ema60_1d_aligned[i]
        bull_power_turning = bull_power_1d_aligned[i] > 0 and (i == start_idx or bull_power_1d_aligned[i-1] <= 0)
        bear_power_turning = bear_power_1d_aligned[i] < 0 and (i == start_idx or bear_power_1d_aligned[i-1] >= 0)
        
        if position == 0:
            # Long: Bear Power turns positive (bullish momentum) in uptrend
            if bull_power_turning and price_above_ema60 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power turns negative (bearish momentum) in downtrend
            elif bear_power_turning and price_below_ema60 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bear Power turns negative or price breaks below EMA60
            if bear_power_1d_aligned[i] < 0 or price_below_ema60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bull Power turns positive or price breaks above EMA60
            if bull_power_1d_aligned[i] > 0 or price_above_ema60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals