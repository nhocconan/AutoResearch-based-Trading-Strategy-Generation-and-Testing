#!/usr/bin/env python3
# 4h_Camarilla_R3S3_Breakout_1dTrend_Volume2
# Hypothesis: Breakout above/below daily Camarilla R3/S3 levels with volume >1.8x 20-bar average and trend filter from 1d EMA50.
# Uses 1d trend for better trend alignment, reducing whipsaw in sideways markets. Designed for 20-40 trades/year on 4h timeframe.
# Works in bull markets (trend-following breakouts) and bear markets (mean reversion at extreme levels).

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume2"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(50) with proper initialization
    ema_50_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[0:50])
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = (close_1d[i] * 2 + ema_50_1d[i-1] * 48) / 50
    
    # Calculate daily Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    daily_range = high_1d - low_1d
    camarilla_R3 = close_1d + daily_range * 1.1 / 2
    camarilla_S3 = close_1d - daily_range * 1.1 / 2
    
    # Align 1d EMA and Camarilla levels to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    
    # Volume filter: 4h volume / 20-period average volume
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
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_R3_aligned[i]) or \
           np.isnan(camarilla_S3_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above Camarilla R3 AND volume confirmation AND bullish trend (price > EMA50)
            if close[i] > camarilla_R3_aligned[i] and volume_ratio[i] > 1.8 and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Camarilla S3 AND volume confirmation AND bearish trend (price < EMA50)
            elif close[i] < camarilla_S3_aligned[i] and volume_ratio[i] > 1.8 and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below Camarilla S3 (reversal signal) or trend turns bearish
            if close[i] < camarilla_S3_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above Camarilla R3 (reversal signal) or trend turns bullish
            if close[i] > camarilla_R3_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals