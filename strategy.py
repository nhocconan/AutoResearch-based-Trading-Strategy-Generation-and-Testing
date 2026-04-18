#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h EMA34 trend filter and volume confirmation.
Donchian breakouts capture momentum, EMA34 filters trend direction, volume confirms institutional participation.
Designed for 15-30 trades/year on 6h timeframe to minimize fee drag. Works in bull markets (buy upper band breaks)
and bear markets (sell lower band breaks) by using 12h EMA34 as trend filter.
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
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h close
    ema_34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        ema_34_12h[33] = np.mean(close_12h[:34])
        for i in range(34, len(close_12h)):
            ema_34_12h[i] = (close_12h[i] * 2 + ema_34_12h[i-1] * 33) / 35
    
    # Align 12h EMA34 to 6h timeframe
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Donchian channels (20-period) on 6h data
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(20, n):
        upper[i] = np.max(high[i-20:i])
        lower[i] = np.min(low[i-20:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation
            if vol_confirmed:
                # Long breakout above upper Donchian band (only if above 12h EMA34)
                if close[i] > upper[i] and close[i] > ema_34_12h_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below lower Donchian band (only if below 12h EMA34)
                elif close[i] < lower[i] and close[i] < ema_34_12h_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price returns to 12h EMA34 or breaks below lower band
            if close[i] < ema_34_12h_aligned[i] or close[i] < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to 12h EMA34 or breaks above upper band
            if close[i] > ema_34_12h_aligned[i] or close[i] > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hEMA34_Volume"
timeframe = "6h"
leverage = 1.0