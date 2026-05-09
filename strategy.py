#!/usr/bin/env python3
# 12h_Camarilla_R3S3_Breakout_1wTrend_Volume
# Hypothesis: Breakout above/below daily Camarilla R3/S3 levels with volume >1.5x 24-bar average and trend filter from 1w EMA20.
# Weekly trend filter reduces whipsaw in sideways markets. Volume filter ensures conviction.
# Designed for 12-30 trades/year on 12h timeframe to avoid fee drag.

name = "12h_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate weekly EMA(20) with proper initialization
    ema_20_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 20:
        ema_20_1w[19] = np.mean(close_1w[0:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = (close_1w[i] * 2 + ema_20_1w[i-1] * 18) / 20
    
    # Align 1w EMA to 12h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    daily_range = high_1d - low_1d
    camarilla_R3 = close_1d + daily_range * 1.1 / 2
    camarilla_S3 = close_1d - daily_range * 1.1 / 2
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume filter: 12h volume / 24-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or \
           np.isnan(camarilla_S3_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above Camarilla R3 AND volume confirmation AND bullish trend (price > weekly EMA20)
            if close[i] > camarilla_R3_aligned[i] and volume_ratio[i] > 1.5 and close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Camarilla S3 AND volume confirmation AND bearish trend (price < weekly EMA20)
            elif close[i] < camarilla_S3_aligned[i] and volume_ratio[i] > 1.5 and close[i] < ema_20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below Camarilla S3 (reversal signal) or trend turns bearish
            if close[i] < camarilla_S3_aligned[i] or close[i] < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above Camarilla R3 (reversal signal) or trend turns bullish
            if close[i] > camarilla_R3_aligned[i] or close[i] > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals