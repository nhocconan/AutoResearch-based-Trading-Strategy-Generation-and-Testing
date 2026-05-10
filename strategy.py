#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
Hypothesis: Price breaks Camarilla R1 (long) or S1 (short) calculated from daily data with 1-day EMA34 trend filter and volume confirmation.
Camarilla levels provide high-probability reversal/breakout points. Daily trend filter ensures alignment with higher timeframe direction.
Volume confirmation filters false breakouts. Works in bull/bear by trading only in direction of daily trend.
Target: 25-35 trades/year (100-140 total) to minimize fee drag.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
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
    
    # 1d data for Camarilla levels, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Camarilla levels from previous day (H, L, C)
    # R1 = C + (H - L) * 1.12
    # S1 = C - (H - L) * 1.12
    camarilla_R1 = np.full(len(close_1d), np.nan)
    camarilla_S1 = np.full(len(close_1d), np.nan)
    
    if len(high_1d) >= 2:
        for i in range(1, len(high_1d)):
            H = high_1d[i-1]
            L = low_1d[i-1]
            C = close_1d[i-1]
            camarilla_R1[i] = C + (H - L) * 1.12
            camarilla_S1[i] = C - (H - L) * 1.12
    
    # 1d EMA34 for trend filter
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    
    # 1d volume SMA20 for volume confirmation
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    
    # Align 1d indicators to 4h (6 bars per day)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Wait for EMA34
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h-equivalent volume from 1d data
        # 1d bar = 6 x 4h bars, so divide by 6 to get average 4h volume
        vol_1d_scaled = vol_sma20_1d_aligned[i] / 6.0
        volume_confirm = volume[i] > 1.5 * vol_1d_scaled
        
        # Trend and price relative to Camarilla levels
        is_uptrend = close[i] > ema34_1d_aligned[i]
        is_downtrend = close[i] < ema34_1d_aligned[i]
        price_above_R1 = close[i] > camarilla_R1_aligned[i]
        price_below_S1 = close[i] < camarilla_S1_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1, in uptrend, with volume
            if price_above_R1 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, in downtrend, with volume
            elif price_below_S1 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below R1 or trend turns down
            if not price_above_R1 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above S1 or trend turns up
            if not price_below_S1 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals