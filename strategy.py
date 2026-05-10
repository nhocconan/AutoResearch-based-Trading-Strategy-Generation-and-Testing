#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend
Hypothesis: Price breaks weekly Camarilla R1 (long) or S1 (short) levels calculated from prior week's range, with 1w EMA200 trend filter and volume confirmation.
Weekly timeframe provides strong trend context for 12h entries, reducing whipsaw in bear markets. Volume confirms breakout strength.
Target: 25-35 trades/year (100-140 total) to minimize fee drag.
"""

name = "12h_Camarilla_R1_S1_Breakout_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly Camarilla levels from prior week: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1 = close_1w + 1.1 * (high_1w - low_1w) / 12
    camarilla_s1 = close_1w - 1.1 * (high_1w - low_1w) / 12
    
    # Weekly EMA200 for trend filter
    ema200_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema200_1w[199] = np.mean(close_1w[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_1w)):
            ema200_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema200_1w[i-1]
    
    # Weekly volume SMA20 for volume confirmation
    vol_sma20_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 20:
        vol_sma20_1w[19] = np.mean(volume_1w[:20])
        for i in range(20, len(df_1w)):
            vol_sma20_1w[i] = (vol_sma20_1w[i-1] * 19 + volume_1w[i]) / 20
    
    # Align weekly indicators to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    vol_sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_sma20_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA200
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_sma20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 2.0x average weekly volume (scaled)
        # 2 weeks in a month, 2*7*2=28 twelve-hour bars per week
        vol_1w_scaled = vol_sma20_1w_aligned[i] / 28.0
        volume_confirm = volume[i] > 2.0 * vol_1w_scaled
        
        # Trend and price relative to weekly Camarilla levels
        is_uptrend = close[i] > ema200_1w_aligned[i]
        is_downtrend = close[i] < ema200_1w_aligned[i]
        price_above_r1 = close[i] > r1_aligned[i]
        price_below_s1 = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly R1, in weekly uptrend, with volume
            if price_above_r1 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S1, in weekly downtrend, with volume
            elif price_below_s1 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below weekly R1 or weekly trend turns down
            if not price_above_r1 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above weekly S1 or weekly trend turns up
            if not price_below_s1 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals