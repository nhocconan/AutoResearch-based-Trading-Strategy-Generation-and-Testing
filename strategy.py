#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout with volume confirmation and 12h EMA trend filter.
Breakouts above/below Donchian(20) channels capture momentum, volume confirms institutional participation,
and 12h EMA filters for trend direction. Designed for 25-35 trades/year to minimize fee drag.
Works in bull markets (buy breakouts above upper band) and bear markets (sell breakdowns below lower band).
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(34) on 12h data
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        ema_12h[33] = np.mean(close_12h[:34])  # simple average for first value
        for i in range(34, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 / (34 + 1)) + (ema_12h[i-1] * (32 / (34 + 1)))
    
    # Align 12h EMA to 4h timeframe
    ema_12h_4h = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
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
    
    start_idx = 20  # need Donchian and volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_12h_4h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 12h EMA
        if position == 0:
            # Only trade in direction of 12h trend
            if vol_confirmed:
                # Long breakout above upper Donchian band
                if close[i] > upper[i] and close[i] > ema_12h_4h[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below lower Donchian band
                elif close[i] < lower[i] and close[i] < ema_12h_4h[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price returns to lower Donchian band or trend reverses
            if close[i] < lower[i] or close[i] < ema_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to upper Donchian band or trend reverses
            if close[i] > upper[i] or close[i] > ema_12h_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0