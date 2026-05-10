#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: Price breaks Camarilla R3/S3 levels calculated from 1d data, with 1w EMA34 trend filter and volume confirmation.
This strategy targets significant breakouts from key intraday pivot levels while ensuring alignment with weekly trend.
Volume confirmation filters false breakouts. The 12h timeframe reduces trade frequency to minimize fee drag.
Designed to work in both bull and bear markets by trading only in direction of weekly trend.
Target: 25-35 trades/year (100-140 total) to stay within optimal range.
"""

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
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
    
    # 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3 = np.full(len(high_1d), np.nan)
    camarilla_s3 = np.full(len(low_1d), np.nan)
    
    for i in range(len(high_1d)):
        if i < 1:  # Need previous day's data
            continue
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        camarilla_r3[i] = prev_close + range_val * 1.1 / 4
        camarilla_s3[i] = prev_close - range_val * 1.1 / 4
    
    # 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on weekly close
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema34_1w[i-1]
    
    # Align 1d Camarilla levels to 12h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Align 1w EMA34 to 12h
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 1  # Need at least one previous day for Camarilla
    
    for i in range(start_idx, n):
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 1d volume (scaled to 12h equivalent)
        # Since 1d = 2 x 12h bars, we scale 1d volume by 0.5 to get equivalent 12h bar volume
        # We'll use a simple volume comparison - current 12h volume vs average 12h volume derived from 1d
        # For simplicity, we use volume > 1.5x the 12h volume from 20 periods ago as proxy
        if i >= 20:
            vol_ma20 = np.mean(volume[i-20:i])
            volume_confirm = volume[i] > 1.5 * vol_ma20
        else:
            volume_confirm = False  # Not enough data for volume confirmation
        
        # Trend and price relative to Camarilla levels
        is_uptrend = close[i] > ema34_1w_aligned[i]
        is_downtrend = close[i] < ema34_1w_aligned[i]
        price_above_r3 = close[i] > camarilla_r3_aligned[i]
        price_below_s3 = close[i] < camarilla_s3_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3, in uptrend, with volume
            if price_above_r3 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, in downtrend, with volume
            elif price_below_s3 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price falls back below R3 or trend turns down
            if not price_above_r3 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price rises back above S3 or trend turns up
            if not price_below_s3 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals