#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h timeframe with 4h/1d trend filter and volume confirmation.
# Uses 4h Supertrend for trend direction, 1d high/low for breakout levels, and volume spike for confirmation.
# Designed to work in both bull and bear markets by following higher timeframe trends.
# Target: 15-37 trades/year to avoid fee drag.

name = "1h_4h_1d_Supertrend_Breakout_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Supertrend (once before loop)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR for Supertrend (10-period)
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.max([high_4h[0] - low_4h[0], np.abs(high_4h[0] - close_4h[0]), np.abs(low_4h[0] - close_4h[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend
    hl2 = (high_4h + low_4h) / 2
    upper = hl2 + (3 * atr)
    lower = hl2 - (3 * atr)
    
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper[0]
    direction[0] = 1
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_4h[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = max(lower[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper[i], supertrend[i-1])
    
    # Align Supertrend and direction to 1h timeframe
    supertrend_1h = align_htf_to_ltf(prices, df_4h, supertrend)
    direction_1h = align_htf_to_ltf(prices, df_4h, direction)
    
    # Get 1d data for breakout levels (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Use previous day's high/low for breakout levels
    prev_high_1d = np.concatenate([[np.nan], high_1d[:-1]])
    prev_low_1d = np.concatenate([[np.nan], low_1d[:-1]])
    
    # Align 1d levels to 1h timeframe
    prev_high_1h = align_htf_to_ltf(prices, df_1d, prev_high_1d)
    prev_low_1h = align_htf_to_ltf(prices, df_1d, prev_low_1d)
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(supertrend_1h[i]) or np.isnan(direction_1h[i]) or \
           np.isnan(prev_high_1h[i]) or np.isnan(prev_low_1h[i]) or \
           np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_24[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: Price breaks above previous day's high with volume and 4h uptrend
            if price > prev_high_1h[i] and volume_confirmed and direction_1h[i] == 1:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below previous day's low with volume and 4h downtrend
            elif price < prev_low_1h[i] and volume_confirmed and direction_1h[i] == -1:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: Price closes below 4h Supertrend
            if price < supertrend_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: Price closes above 4h Supertrend
            if price > supertrend_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals