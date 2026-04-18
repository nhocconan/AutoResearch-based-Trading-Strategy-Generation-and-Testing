#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA(34) trend filter and volume confirmation.
Long when price breaks above upper Donchian channel with price above 12h EMA(34) and volume spike.
Short when price breaks below lower Donchian channel with price below 12h EMA(34) and volume spike.
Exit when price crosses back through the opposite Donchian band or volatility drops.
Designed for 25-35 trades/year to minimize fee drag while capturing trends in both bull and bear markets.
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
    
    # Get 12h data for EMA(34)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(34) on 12h
    ema_34_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 34:
        ema_34_12h[33] = np.mean(close_12h[:34])
        for i in range(34, len(close_12h)):
            ema_34_12h[i] = (close_12h[i] * 2 / 35) + ema_34_12h[i-1] * (33 / 35)
    
    # Align EMA to 4h timeframe
    ema_34_12h_4h = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Donchian channels (20-period)
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(20, n):
        upper_channel[i] = np.max(high[i-20:i])
        lower_channel[i] = np.min(low[i-20:i])
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need Donchian and volume MA calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_4h[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian, above 12h EMA, volume confirmation
            if close[i] > upper_channel[i] and close[i] > ema_34_12h_4h[i] and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, below 12h EMA, volume confirmation
            elif close[i] < lower_channel[i] and close[i] < ema_34_12h_4h[i] and vol_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower Donchian channel or volume drops
            if close[i] < lower_channel[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper Donchian channel or volume drops
            if close[i] > upper_channel[i] or volume[i] < vol_ma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA34_Volume"
timeframe = "4h"
leverage = 1.0