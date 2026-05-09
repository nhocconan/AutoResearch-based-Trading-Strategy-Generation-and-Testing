#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend
# Hypothesis: Buy when price breaks above Camarilla R1 level and sell when price breaks below S1 level on 12h timeframe, 
# filtered by 1w EMA34 trend direction (bullish when price > EMA34, bearish when price < EMA34). 
# Uses volume confirmation (volume > 1.8x average) to filter low-conviction moves. 
# Designed for 15-35 trades per year on 12h timeframe. Works in bull markets via breakouts in uptrend 
# and in bear markets via breakdowns in downtrend.

name = "12h_Camarilla_R1_S1_Breakout_1wTrend"
timeframe = "12h"
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
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34)
    ema_34_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema_34_1w[33] = np.mean(close_1w[0:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = (close_1w[i] * 2 + ema_34_1w[i-1] * 32) / 34
    
    # Align 1w EMA to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels using previous week's OHLC
    # Camarilla levels are calculated from previous week's range
    # R1 = C + (H-L) * 1.1/12
    # S1 = C - (H-L) * 1.1/12
    # We need to shift by 1 to use previous week's data
    
    # Calculate weekly range from 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_prev = np.roll(close_1w, 1)  # Previous week's close
    close_1w_prev[0] = np.nan  # First week has no previous
    
    # Calculate Camarilla levels for each week
    camarilla_R1 = np.full_like(close_1w, np.nan)
    camarilla_S1 = np.full_like(close_1w, np.nan)
    
    valid = ~np.isnan(close_1w_prev)
    camarilla_R1[valid] = close_1w_prev[valid] + (high_1w[valid] - low_1w[valid]) * 1.1 / 12
    camarilla_S1[valid] = close_1w_prev[valid] - (high_1w[valid] - low_1w[valid]) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S1)
    
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
    
    start_idx = max(20, 1)  # Ensure volume MA is ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or \
           np.isnan(camarilla_S1_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above R1 AND volume confirmation AND bullish trend (price > EMA)
            if close[i] > camarilla_R1_aligned[i] and volume_ratio[i] > 1.8 and close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below S1 AND volume confirmation AND bearish trend (price < EMA)
            elif close[i] < camarilla_S1_aligned[i] and volume_ratio[i] > 1.8 and close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below S1 (reversal signal) or trend turns bearish
            if close[i] < camarilla_S1_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above R1 (reversal signal) or trend turns bullish
            if close[i] > camarilla_R1_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals