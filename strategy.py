#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Breakout above/below daily Camarilla R1/S1 levels with volume >2x 30-bar average and trend filter from 1d EMA34.
# Uses daily Camarilla pivot levels as strong intraday support/resistance. In uptrend (price > EMA34), buy breakout above R1; in downtrend (price < EMA34), sell breakdown below S1.
# Volume filter ensures only high-conviction moves trigger entries. Designed for 12h timeframe to capture 15-30 trades/year.

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for EMA trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34) with proper initialization
    ema_34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[0:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2 + ema_34_1d[i-1] * 32) / 34
    
    # Align 1d EMA to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily Camarilla pivot levels
    # P = (H+L+C)/3, R1 = C + (H-L)*1.1/2, S1 = C - (H-L)*1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    daily_range = high_1d - low_1d
    camarilla_R1 = close_1d + daily_range * 1.1 / 2
    camarilla_S1 = close_1d - daily_range * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume filter: 12h volume / 30-period average volume
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 30:
        vol_ma[29] = np.mean(volume[0:30])
        for i in range(30, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 29 + volume[i]) / 30
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 30)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or \
           np.isnan(camarilla_S1_aligned[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price breaks above daily Camarilla R1 AND volume confirmation AND bullish trend (price > EMA34)
            if close[i] > camarilla_R1_aligned[i] and volume_ratio[i] > 2.0 and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below daily Camarilla S1 AND volume confirmation AND bearish trend (price < EMA34)
            elif close[i] < camarilla_S1_aligned[i] and volume_ratio[i] > 2.0 and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below daily Camarilla S1 (reversal signal) or trend turns bearish
            if close[i] < camarilla_S1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above daily Camarilla R1 (reversal signal) or trend turns bullish
            if close[i] > camarilla_R1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals