#!/usr/bin/env python3
"""
12h_1d_Pivot_R1_S1_Breakout_Volume
Hypothesis: Trade 12h breakouts above Camarilla R1 or below S1 levels derived from 1d candles, 
with volume confirmation and a 1w EMA filter to avoid counter-trend trades. 
Camarilla levels provide institutional support/resistance; volume ensures momentum; 
weekly EMA ensures trend alignment. Works in bull via long breakouts and in bear via short breakdowns. 
Target: 12-37 trades/year by requiring confluence of level break, volume spike, and trend filter.
"""

import numpy as np
import pandas as pd
from math import ceil
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar: R1, S1
    camarilla_R1 = np.full_like(close_1d, np.nan)
    camarilla_S1 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i < 1:  # Need previous day
            continue
        H = high_1d[i-1]
        L = low_1d[i-1]
        C = close_1d[i-1]
        camarilla_R1[i] = C + (H - L) * 1.1 / 12
        camarilla_S1[i] = C - (H - L) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w EMA34
    ema_period = 34
    ema_1w = np.full_like(close_1w, np.nan)
    
    if len(close_1w) >= ema_period:
        # Use pandas EMA for efficiency and correctness
        ema_series = pd.Series(close_1w).ewm(span=ema_period, adjust=False).values
        ema_1w[:] = ema_series
    
    # Align 1w EMA34 to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 2.0x 20-period average (higher threshold to reduce trades)
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all warmup periods
    start_idx = max(vol_period, 1) + 1  # +1 for Camarilla needing prior day
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: at least 2x average volume
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R1, above weekly EMA, with volume
            if close[i] > camarilla_R1_aligned[i] and close[i] > ema_1w_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S1, below weekly EMA, with volume
            elif close[i] < camarilla_S1_aligned[i] and close[i] < ema_1w_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Camarilla S1 or below weekly EMA
            if close[i] < camarilla_S1_aligned[i] or close[i] < ema_1w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above Camarilla R1 or above weekly EMA
            if close[i] > camarilla_R1_aligned[i] or close[i] > ema_1w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Pivot_R1_S1_Breakout_Volume"
timeframe = "12h"
leverage = 1.0