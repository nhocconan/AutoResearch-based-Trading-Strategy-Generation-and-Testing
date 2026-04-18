#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_TrendFilter
Hypothesis: Trade Camarilla pivot breakouts at R1 (long) and S1 (short) with volume confirmation and 1d EMA34 trend filter. Camarilla levels provide high-probability reversal/breakout points. Only take longs when price > 1d EMA34 (uptrend) and shorts when price < 1d EMA34 (downtrend) to avoid counter-trend trades. Volume > 1.5x 24-period average confirms breakout strength. Targets 20-30 trades/year via strict pivot breaks + trend + volume filters. Works in bull/bear by following higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and EMA34
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each 1d bar: R1, S1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_R1 = np.full_like(close_1d, np.nan)
    camarilla_S1 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i >= 0:  # Need at least one bar
            rng = high_1d[i] - low_1d[i]
            camarilla_R1[i] = close_1d[i] + rng * 1.1 / 12
            camarilla_S1[i] = close_1d[i] - rng * 1.1 / 12
    
    # Calculate EMA34 on 1d close
    ema_period = 34
    ema_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period-1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (ema_period + 1)) + (ema_1d[i-1] * (ema_period - 1) / (ema_period + 1))
    
    # Align 1d indicators to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 24-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(vol_period, 1)  # Need volume MA and at least 1 bar for crossover
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above R1 + uptrend (price > EMA34) + volume
            if close[i] > camarilla_R1_aligned[i] and close[i] > ema_1d_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + downtrend (price < EMA34) + volume
            elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_1d_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or trend turns down (price < EMA34)
            if close[i] < camarilla_S1_aligned[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or trend turns up (price > EMA34)
            if close[i] > camarilla_R1_aligned[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0