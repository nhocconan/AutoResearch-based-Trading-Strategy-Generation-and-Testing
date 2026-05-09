#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS
# Hypothesis: Combines 12h EMA50 trend with 1d Camarilla R1/S1 breakout and volume confirmation.
# Uses 12h EMA50 for trend direction, 1d Camarilla R1/S1 for breakout levels, and volume > 2x average for confirmation.
# Designed to work in both bull and bear markets by only taking trades in the direction of the 12h trend.
# Target: 20-35 trades/year per symbol with disciplined risk.

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
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
    
    # Get 12h data for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50
    ema_12h_50 = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 50:
        ema_12h_50[49] = np.mean(close_12h[0:50])
        for i in range(50, len(close_12h)):
            ema_12h_50[i] = (close_12h[i] * 2 + ema_12h_50[i-1] * 49) / 51
    
    # Align 12h EMA50 to 4h timeframe
    ema_12h_50_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_50)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    daily_range = high_1d - low_1d
    camarilla_R1 = close_1d + daily_range * 1.1 / 12
    camarilla_S1 = close_1d - daily_range * 1.1 / 12
    
    # Align daily Camarilla levels to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
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
    
    start_idx = max(50, 1)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_12h_50_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or \
           np.isnan(camarilla_S1_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend direction from 12h EMA50
        trend_up = close[i] > ema_12h_50_aligned[i]
        trend_down = close[i] < ema_12h_50_aligned[i]
        
        if position == 0:
            # Enter long: Price breaks above Camarilla R1 AND trend up AND volume confirmation
            if close[i] > camarilla_R1_aligned[i] and trend_up and volume_ratio[i] > 2.0:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Camarilla S1 AND trend down AND volume confirmation
            elif close[i] < camarilla_S1_aligned[i] and trend_down and volume_ratio[i] > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions: Price breaks below Camarilla S1 OR trend turns down
            if close[i] < camarilla_S1_aligned[i] or not trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: Price breaks above Camarilla R1 OR trend turns up
            if close[i] > camarilla_R1_aligned[i] or not trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals