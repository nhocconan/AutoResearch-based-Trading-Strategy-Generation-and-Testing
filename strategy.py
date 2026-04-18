#!/usr/bin/env python3
"""
12h_1d_Weekly_EMA34_Filter_Support_Resistance_Breakout
Hypothesis: Trade 12h breakouts above 1d resistance (previous day high) or below 1d support (previous day low) only when aligned with weekly EMA(34) trend and confirmed by volume > 2x 24-period average. This captures institutional breakout attempts while filtering false signals. Weekly trend ensures directional bias, volume confirms institutional participation. Designed for low trade frequency (<25/year) to minimize fee drag in ranging markets like 2025 BTC/ETH.
"""

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
    
    # Get 1d data for support/resistance levels
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Previous 1d high/low for support/resistance (completed day)
    prev_high_1d = np.roll(df_1d['high'].values, 1)
    prev_low_1d = np.roll(df_1d['low'].values, 1)
    prev_high_1d[0] = df_1d['high'].values[0]
    prev_low_1d[0] = df_1d['low'].values[0]
    
    # Weekly EMA(34) trend filter
    close_1w = df_1w['close'].values
    ema_period = 34
    ema_1w = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2 / (ema_period + 1)) + (ema_1w[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align 1d support/resistance and weekly EMA to 12h timeframe
    resistance = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    support = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 2x 24-period average (strict filter for low frequency)
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, vol_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(resistance[i]) or np.isnan(support[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation (>2x average for institutional participation)
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: break above resistance with volume and above weekly EMA
            if close[i] > resistance[i] and vol_confirm and close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below support with volume and below weekly EMA
            elif close[i] < support[i] and vol_confirm and close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes back below support or below weekly EMA
            if close[i] < support[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes back above resistance or above weekly EMA
            if close[i] > resistance[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Weekly_EMA34_Filter_Support_Resistance_Breakout"
timeframe = "12h"
leverage = 1.0