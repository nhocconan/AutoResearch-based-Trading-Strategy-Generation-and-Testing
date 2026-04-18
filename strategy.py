#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA34 trend filter.
Long when price breaks above upper Donchian channel in uptrend (12h EMA34 rising).
Short when price breaks below lower Donchian channel in downtrend (12h EMA34 falling).
Volume confirmation filters breakouts. Designed for 20-30 trades/year to minimize fee drag.
Works in bull markets (buy breakouts) and bear markets (sell breakdowns).
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
    
    # Align 12h EMA34 to 4h timeframe
    ema_34_12h_4h = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate EMA34 slope (rising/falling)
    ema_slope = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(ema_34_12h_4h[i]) and not np.isnan(ema_34_12h_4h[i-1]):
            ema_slope[i] = ema_34_12h_4h[i] - ema_34_12h_4h[i-1]
    
    # Calculate Donchian channels (20-period) on 4h data
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(20, n):
        donch_high[i] = np.max(high[i-20:i])
        donch_low[i] = np.min(low[i-20:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_slope[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3 * 20-period average
        vol_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation
            if vol_confirmed:
                # Long breakout above upper Donchian in uptrend (rising EMA)
                if close[i] > donch_high[i] and ema_slope[i] > 0:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown below lower Donchian in downtrend (falling EMA)
                elif close[i] < donch_low[i] and ema_slope[i] < 0:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price returns to midpoint of Donchian channel
            mid = (donch_high[i] + donch_low[i]) / 2
            if close[i] <= mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to midpoint of Donchian channel
            mid = (donch_high[i] + donch_low[i]) / 2
            if close[i] >= mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0