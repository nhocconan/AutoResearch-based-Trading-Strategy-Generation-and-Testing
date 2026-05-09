#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend
# Hypothesis: Buy when price breaks above Camarilla R1 and sell when price breaks below S1 on 12h timeframe,
# filtered by 1d EMA34 trend (price > EMA34 bullish, price < EMA34 bearish) and volume > 1.5x average.
# Designed for 12-37 trades/year on 12h timeframe with strict entry conditions to minimize fee drag.
# Works in bull markets via breakouts in uptrend and bear markets via breakdowns in downtrend.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA(34) with proper initialization
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 + ema_34_1d[i-1] * 32) / 34
    
    # Align 1d EMA to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily range from 1d data for Camarilla levels
    close_1d_prev = np.roll(close_1d, 1)
    close_1d_prev[0] = np.nan
    
    # Calculate Camarilla levels for each day
    camarilla_R1 = np.full_like(close_1d, np.nan)
    camarilla_S1 = np.full_like(close_1d, np.nan)
    
    valid = ~np.isnan(close_1d_prev)
    camarilla_R1[valid] = close_1d_prev[valid] + (high_1d[valid] - low_1d[valid]) * 1.1 / 12
    camarilla_S1[valid] = close_1d_prev[valid] - (high_1d[valid] - low_1d[valid]) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume filter: 12h volume / 20-period average volume
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
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or \
           np.isnan(camarilla_S1_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above R1 AND volume confirmation AND bullish trend (price > EMA)
            if close[i] > camarilla_R1_aligned[i] and volume_ratio[i] > 1.5 and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below S1 AND volume confirmation AND bearish trend (price < EMA)
            elif close[i] < camarilla_S1_aligned[i] and volume_ratio[i] > 1.5 and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below S1 (reversal signal) or trend turns bearish
            if close[i] < camarilla_S1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above R1 (reversal signal) or trend turns bullish
            if close[i] > camarilla_R1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals