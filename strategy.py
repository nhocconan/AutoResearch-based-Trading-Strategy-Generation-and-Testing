#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA trend filter.
Long when price breaks above Donchian upper band with volume > 1.5x average and 12h EMA34 rising.
Short when price breaks below Donchian lower band with volume > 1.5x average and 12h EMA34 falling.
Exit on opposite Donchian break or volume dry-up. Uses discrete position sizing (0.25) to minimize fee drag.
Designed to capture trending moves in both bull and bear markets with minimal whipsaws.
Target: 20-50 trades/year per symbol.
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
    
    # Calculate Donchian channels (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 0)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_12h_rising = np.gradient(ema_12h) > 0  # True when EMA is rising
    ema_12h_falling = np.gradient(ema_12h) < 0  # True when EMA is falling
    
    # Align 12h EMA signals to 4h
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_falling)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(20, 34)  # Donchian and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(ema_rising_aligned[i]) or 
            np.isnan(ema_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: price breaks above Donchian high + volume spike + 12h EMA rising
            if (close[i] > donchian_high[i] and 
                vol_ratio[i] > 1.5 and 
                ema_rising_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low + volume spike + 12h EMA falling
            elif (close[i] < donchian_low[i] and 
                  vol_ratio[i] > 1.5 and 
                  ema_falling_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR volume dry-up
            if (close[i] < donchian_low[i] or vol_ratio[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR volume dry-up
            if (close[i] > donchian_high[i] or vol_ratio[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Volume_EMA12h"
timeframe = "4h"
leverage = 1.0