#!/usr/bin/env python3
"""
4h_4WkHigh_Low_Breakout_12hTrend_v2
Hypothesis: Price breaks the 4-week high (long) or low (short) calculated from 12h data, with 12h EMA50 trend filter and volume confirmation.
Breakouts from multi-week extremes capture sustained momentum, while 12h trend filter ensures alignment with intermediate-term direction.
Volume confirmation filters false breakouts. Works in bull/bear by trading only in direction of 12h trend.
Target: 20-40 trades/year (80-160 total) to minimize fee drift.
"""

name = "4h_4WkHigh_Low_Breakout_12hTrend_v2"
timeframe = "4h"
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
    
    # 12h data for multi-week extremes and trend
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 4-week high/low from 12h data (20 periods = 10 days ≈ 2 weeks, 40 = 4 weeks)
    lookback = 40
    high_4wk = np.full(len(high_12h), np.nan)
    low_4wk = np.full(len(low_12h), np.nan)
    
    if len(high_12h) >= lookback:
        for i in range(lookback, len(high_12h)):
            high_4wk[i] = np.max(high_12h[i-lookback:i])
            low_4wk[i] = np.min(low_12h[i-lookback:i])
    
    # 12h EMA50 for trend filter
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema50_12h[i-1]
    
    # 12h volume SMA20 for volume confirmation
    vol_sma20_12h = np.full(len(volume_12h), np.nan)
    if len(volume_12h) >= 20:
        vol_sma20_12h[19] = np.mean(volume_12h[:20])
        for i in range(20, len(volume_12h)):
            vol_sma20_12h[i] = (vol_sma20_12h[i-1] * 19 + volume_12h[i]) / 20
    
    # Align 12h indicators to 4h
    high_4wk_aligned = align_htf_to_ltf(prices, df_12h, high_4wk)
    low_4wk_aligned = align_htf_to_ltf(prices, df_12h, low_4wk)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_sma20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_sma20_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(high_4wk_aligned[i]) or np.isnan(low_4wk_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_sma20_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 12h volume (scaled)
        # 12h bar = 3 x 4h bars, so to compare 4h volume to average 12h volume, we need to scale
        vol_12h_scaled = vol_sma20_12h_aligned[i] / 3.0  # Average 4h-equivalent volume from 12h data
        volume_confirm = volume[i] > 1.5 * vol_12h_scaled
        
        # Trend and price relative to 4-week levels
        is_uptrend = close[i] > ema50_12h_aligned[i]
        is_downtrend = close[i] < ema50_12h_aligned[i]
        price_above_4wk_high = close[i] > high_4wk_aligned[i]
        price_below_4wk_low = close[i] < low_4wk_aligned[i]
        
        if position == 0:
            # Long: price breaks above 4-week high, in uptrend, with volume
            if price_above_4wk_high and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 4-week low, in downtrend, with volume
            elif price_below_4wk_low and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below 4-week high or trend turns down
            if not price_above_4wk_high or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above 4-week low or trend turns up
            if not price_below_4wk_low or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals