#!/usr/bin/env python3
"""
4h_4H_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS
Hypothesis: Price breaks above/below daily Camarilla R1/S1 levels with 12h EMA50 trend filter and volume confirmation. 
Camarilla R1/S1 provide tight support/resistance in ranging markets, while the 12h EMA50 ensures alignment with medium-term momentum. 
Volume filters false breakouts. Works in bull/bear by trading only in direction of 12h trend. 
Target: 20-40 trades/year (80-160 total) to minimize fee drag.
"""

name = "4h_4H_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeS"
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
    
    # Daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each daily bar
    # R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema50_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 50:
        ema50_12h[49] = np.mean(close_12h[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_12h)):
            ema50_12h[i] = alpha * close_12h[i] + (1 - alpha) * ema50_12h[i-1]
    
    # Daily volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    
    # Align all indicators to 4h timeframe
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average daily volume (scaled)
        # 1d = 6 x 4h bars, so scale daily volume to 4h equivalent
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 6.0  # Average 4h-equivalent volume from 1d data
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend and price relative to Camarilla levels
        is_uptrend = close[i] > ema50_12h_aligned[i]
        is_downtrend = close[i] < ema50_12h_aligned[i]
        price_above_r1 = close[i] > r1_1d_aligned[i]
        price_below_s1 = close[i] < s1_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1, in uptrend, with volume
            if price_above_r1 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, in downtrend, with volume
            elif price_below_s1 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below R1 or trend turns down
            if not price_above_r1 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above S1 or trend turns up
            if not price_below_s1 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals