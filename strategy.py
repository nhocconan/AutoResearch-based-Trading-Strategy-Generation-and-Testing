#!/usr/bin/env python3
# 4h_12h_Camarilla_R3_S3_Breakout_12hTrend_Volume
# Hypothesis: Breakout above/below 12h Camarilla R3/S3 levels with volume >1.8x 20-bar average and trend filter from 12h EMA50.
# Camarilla levels are strong S/R. Trend filter ensures directional bias. Volume confirms conviction.
# Designed for 20-40 trades/year on 4h timeframe.

name = "4h_12h_Camarilla_R3_S3_Breakout_12hTrend_Volume"
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
    
    # Get 12h data for trend, Camarilla, and volume filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h EMA(50) for trend filter
    ema_50_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_50_12h[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema_50_12h[i] = (close_12h[i] * 2 + ema_50_12h[i-1] * 48) / 50
    
    # Calculate 12h Camarilla levels: R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    # Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # Using R3/S3 as breakout levels
    hl_12h = high_12h - low_12h
    camarilla_R3_12h = close_12h + hl_12h * 1.1 / 4
    camarilla_S3_12h = close_12h - hl_12h * 1.1 / 4
    
    # Volume filter: 12h volume / 20-period average volume
    vol_ma_12h = np.full_like(volume_12h, np.nan)
    if len(volume_12h) >= 20:
        vol_ma_12h[19] = np.mean(volume_12h[0:20])
        for i in range(20, len(volume_12h)):
            vol_ma_12h[i] = (vol_ma_12h[i-1] * 19 + volume_12h[i]) / 20
    
    volume_ratio_12h = np.full_like(volume_12h, np.nan)
    valid_vol = (~np.isnan(vol_ma_12h)) & (vol_ma_12h != 0)
    volume_ratio_12h[valid_vol] = volume_12h[valid_vol] / vol_ma_12h[valid_vol]
    
    # Align all 12h indicators to 4h timeframe
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    camarilla_R3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R3_12h)
    camarilla_S3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S3_12h)
    volume_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_ratio_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_12h_aligned[i]) or np.isnan(camarilla_R3_12h_aligned[i]) or \
           np.isnan(camarilla_S3_12h_aligned[i]) or np.isnan(volume_ratio_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above Camarilla R3 AND volume confirmation AND bullish trend (price > EMA50)
            if close[i] > camarilla_R3_12h_aligned[i] and volume_ratio_12h_aligned[i] > 1.8 and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Camarilla S3 AND volume confirmation AND bearish trend (price < EMA50)
            elif close[i] < camarilla_S3_12h_aligned[i] and volume_ratio_12h_aligned[i] > 1.8 and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below Camarilla S3 (reversal signal) or trend turns bearish
            if close[i] < camarilla_S3_12h_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above Camarilla R3 (reversal signal) or trend turns bullish
            if close[i] > camarilla_R3_12h_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals