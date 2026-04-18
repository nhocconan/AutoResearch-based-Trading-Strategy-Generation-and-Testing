#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_Volume_And_HTF_EMA34
Hypothesis: Trade Camarilla pivot (R1/S1) breakouts on 4h with volume confirmation and 12h EMA34 trend filter.
Long when price breaks above R1 with volume > 1.5x average and 12h EMA34 rising.
Short when price breaks below S1 with volume > 1.5x average and 12h EMA34 falling.
Uses Camarilla levels from prior 1d for structure, volume for conviction, and 12h EMA for trend filter.
Designed for 20-40 trades/year to avoid fee drag. Works in bull/bear by following higher timeframe trend.
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
    
    # Get 1d data for Camarilla pivot levels (prior day)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels for each day: based on prior day's H, L, C
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_R1 = np.full_like(close_1d, np.nan)
    camarilla_S1 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_R1[i] = np.nan
            camarilla_S1[i] = np.nan
        else:
            H = high_1d[i-1]
            L = low_1d[i-1]
            C = close_1d[i-1]
            camarilla_R1[i] = C + (H - L) * 1.1 / 12
            camarilla_S1[i] = C - (H - L) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # EMA34 on 12h
    ema_period = 34
    ema_12h = np.full_like(close_12h, np.nan)
    
    if len(close_12h) >= ema_period:
        multiplier = 2 / (ema_period + 1)
        ema_12h[ema_period-1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * multiplier) + (ema_12h[i-1] * (1 - multiplier))
    
    # Align 12h EMA34 to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, vol_period)  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # EMA trend: rising if current > previous, falling if current < previous
        ema_rising = i > 0 and not np.isnan(ema_12h_aligned[i-1]) and ema_12h_aligned[i] > ema_12h_aligned[i-1]
        ema_falling = i > 0 and not np.isnan(ema_12h_aligned[i-1]) and ema_12h_aligned[i] < ema_12h_aligned[i-1]
        
        if position == 0:
            # Long: price breaks above R1 + volume + EMA rising
            if close[i] > camarilla_R1_aligned[i] and vol_confirm and ema_rising:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume + EMA falling
            elif close[i] < camarilla_S1_aligned[i] and vol_confirm and ema_falling:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 or EMA falls
            if close[i] < camarilla_S1_aligned[i] or (i > 0 and not np.isnan(ema_12h_aligned[i-1]) and ema_12h_aligned[i] < ema_12h_aligned[i-1]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 or EMA rises
            if close[i] > camarilla_R1_aligned[i] or (i > 0 and not np.isnan(ema_12h_aligned[i-1]) and ema_12h_aligned[i] > ema_12h_aligned[i-1]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_Volume_And_HTF_EMA34"
timeframe = "4h"
leverage = 1.0