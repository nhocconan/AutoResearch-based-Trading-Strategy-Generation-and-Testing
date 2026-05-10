#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Camarilla pivot breakout on 1d with weekly trend filter and volume confirmation.
Uses 1w EMA34 as primary trend filter - trades only in direction of weekly trend.
Camarilla R1/S1 levels provide institutional-grade support/resistance for breakouts.
Volume confirmation ensures breakouts have institutional participation.
Designed for low frequency (15-25 trades/year) to minimize fee drag in 2025 bear market.
Works in bull markets (breakouts with volume) and bear markets (fades at pivot levels).
Target: 60-100 total trades over 4 years (15-25/year).
"""

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Weekly EMA34 for trend filter (primary trend direction)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema34_1w[i-1]
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Daily Camarilla pivot levels (R1, S1)
    # Calculate from previous day's OHLC
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close[0] = np.nan  # First bar has no previous
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.0 / 12.0)
    s1 = pivot - (range_hl * 1.0 / 12.0)
    
    # Daily volume SMA20 for confirmation
    vol_sma20 = np.full(n, np.nan)
    if n >= 20:
        vol_sma20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_sma20[i] = (vol_sma20[i-1] * 19 + volume[i]) / 20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # warmup for weekly EMA and volume
    
    for i in range(start_idx, n):
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_sma20[i]) or np.isnan(pivot[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = volume[i] > 1.5 * vol_sma20[i]
        
        if position == 0:
            # Long: Price breaks above R1 with weekly uptrend and volume
            if close[i] > r1[i] and close[i] > ema34_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with weekly downtrend and volume
            elif close[i] < s1[i] and close[i] < ema34_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price returns to pivot or weekly trend turns down
            if close[i] < pivot[i] or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price returns to pivot or weekly trend turns up
            if close[i] > pivot[i] or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals