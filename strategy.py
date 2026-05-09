#!/usr/bin/env python3
# 1h_Camarilla_4h1d_Trend_Signal
# Hypothesis: Uses 4h EMA50 and 1d EMA200 for trend direction (long only when above both, short only when below both).
# Entry timing on 1h: price pulls back to 4h EMA21 during strong trend. Uses volume confirmation (1.5x 20-bar average).
# Designed to capture trend continuation moves with low frequency to avoid fee drag. Works in bull via longs, bear via shorts.
# Target: 15-30 trades/year per symbol.

name = "1h_Camarilla_4h1d_Trend_Signal"
timeframe = "1h"
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
    
    # Get 4h data for EMA21 and EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h EMA21
    ema21_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 21:
        ema21_4h[20] = np.mean(close_4h[0:21])
        for i in range(21, len(close_4h)):
            ema21_4h[i] = (close_4h[i] * 2 + ema21_4h[i-1] * 19) / 21
    
    # Calculate 4h EMA50
    ema50_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 50:
        ema50_4h[49] = np.mean(close_4h[0:50])
        for i in range(50, len(close_4h)):
            ema50_4h[i] = (close_4h[i] * 2 + ema50_4h[i-1] * 48) / 50
    
    # Get 1d data for EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA200
    ema200_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[0:200])
        for i in range(200, len(close_1d)):
            ema200_1d[i] = (close_1d[i] * 2 + ema200_1d[i-1] * 198) / 200
    
    # Align HTF indicators to 1h timeframe
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume filter: 1h volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # wait for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema21_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or \
           np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend condition: 4h EMA50 > 1d EMA200 for long bias, < for short bias
        long_bias = ema50_4h_aligned[i] > ema200_1d_aligned[i]
        short_bias = ema50_4h_aligned[i] < ema200_1d_aligned[i]
        
        if position == 0:
            # Enter long: price pulls back to 4h EMA21 (within 0.5%) during strong uptrend + volume
            if long_bias and close[i] >= ema21_4h_aligned[i] * 0.995 and close[i] <= ema21_4h_aligned[i] * 1.005 and volume_ratio[i] > 1.5:
                signals[i] = 0.20
                position = 1
            # Enter short: price pulls back to 4h EMA21 (within 0.5%) during strong downtrend + volume
            elif short_bias and close[i] >= ema21_4h_aligned[i] * 0.995 and close[i] <= ema21_4h_aligned[i] * 1.005 and volume_ratio[i] > 1.5:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: trend breaks (4h EMA50 < 1d EMA200) or price moves 2% above entry
            if not long_bias or close[i] > ema21_4h_aligned[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: trend breaks (4h EMA50 > 1d EMA200) or price moves 2% below entry
            if not short_bias or close[i] < ema21_4h_aligned[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals