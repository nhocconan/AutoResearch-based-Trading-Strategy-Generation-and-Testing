#!/usr/bin/env python3
# 6h_ElderRay_BullBearPower_1dTrend_Volume
# Hypothesis: Elder Ray Index (Bull Power = High - EMA13, Bear Power = EMA13 - Low) with 1d EMA34 trend filter and volume spike confirmation.
# In bull market (price > 1d EMA34), look for Bull Power > 0 and rising; in bear market (price < 1d EMA34), look for Bear Power > 0 and rising.
# Volume spike (>2x 20-period average) confirms strength of the move.
# Designed for low trade frequency (<40/year) to minimize fee drift in 6h timeframe.
# Works in both bull and bear markets by following the daily trend direction.

name = "6h_ElderRay_BullBearPower_1dTrend_Volume"
timeframe = "6h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema13 = np.full_like(close, np.nan)
    if len(close) >= 13:
        ema13[12] = np.mean(close[0:13])
        for i in range(13, len(close)):
            ema13[i] = (ema13[i-1] * 12 + close[i]) / 13
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (ema_34_1d[i-1] * 33 + close_1d[i]) / 34
    
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter: current volume / 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Ensure volume MA and EMA13 are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0 AND rising AND uptrend (price > EMA34) AND volume spike
            if (bull_power[i] > 0 and 
                i > start_idx and bull_power[i] > bull_power[i-1] and
                close[i] > ema_34_1d_aligned[i] and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power > 0 AND rising AND downtrend (price < EMA34) AND volume spike
            elif (bear_power[i] > 0 and 
                  i > start_idx and bear_power[i] > bear_power[i-1] and
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 OR trend reversal (price < EMA34)
            if bull_power[i] <= 0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power <= 0 OR trend reversal (price > EMA34)
            if bear_power[i] <= 0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals