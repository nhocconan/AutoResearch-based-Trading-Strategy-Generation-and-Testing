#!/usr/bin/env python3
# 6h_ElderRay_BullPower_BearPower_1dTrend
# Hypothesis: Elder Ray Index (Bull Power/Bear Power) with 13-period EMA on 6h timeframe, 
# filtered by 1d EMA34 trend direction. Long when Bull Power > 0 and price > 1d EMA34,
# short when Bear Power < 0 and price < 1d EMA34. Uses volume confirmation (volume > 1.5x average)
# to filter low-conviction moves. Designed for 50-150 trades over 4 years on 6h timeframe.
# Works in bull markets via Bull Power strength and in bear markets via Bear Power weakness.

name = "6h_ElderRay_BullPower_BearPower_1dTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34)
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 + ema_34_1d[i-1] * 32) / 34
    
    # Align 1d EMA to 6h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate EMA(13) for Elder Ray on 6h
    ema_13 = np.full(n, np.nan)
    if n >= 13:
        ema_13[12] = np.mean(close[0:13])
        for i in range(13, n):
            ema_13[i] = (close[i] * 2 + ema_13[i-1] * 11) / 13
    
    # Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume filter: 6h volume / 20-period average volume
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
    
    start_idx = max(20, 13)  # Ensure volume MA and EMA(13) are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power[i]) or \
           np.isnan(bear_power[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bull Power > 0 AND volume confirmation AND price > 1d EMA
            if bull_power[i] > 0 and volume_ratio[i] > 1.5 and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 AND volume confirmation AND price < 1d EMA
            elif bear_power[i] < 0 and volume_ratio[i] > 1.5 and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power <= 0 (loss of bullish momentum) or trend reversal
            if bull_power[i] <= 0 or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power >= 0 (loss of bearish momentum) or trend reversal
            if bear_power[i] >= 0 or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals