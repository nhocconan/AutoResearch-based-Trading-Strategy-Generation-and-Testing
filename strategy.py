#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1d_1w_ema_confluence_v1
# Combines 6h EMA trend with 1d and 1w EMA confluence for trend strength.
# Goes long when 6h EMA > 1d EMA > 1w EMA with volume confirmation.
# Goes short when 6h EMA < 1d EMA < 1w EMA with volume confirmation.
# Uses volume > 1.5x 20-period average to confirm institutional participation.
# Designed to work in both bull and bear markets by requiring multi-timeframe alignment.
# Target: 20-30 trades/year per symbol (~80-120 total over 4 years).
name = "6h_1d_1w_ema_confluence_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate EMA(21) on each timeframe
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    
    ema_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 6h EMA(21)
    ema_6h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(21, n):  # start after EMA warmup
        # Skip if any EMA not ready
        if (np.isnan(ema_6h[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Check volume filter
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long condition: 6h EMA > 1d EMA > 1w EMA (bullish alignment)
        if (ema_6h[i] > ema_1d_aligned[i] > ema_1w_aligned[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short condition: 6h EMA < 1d EMA < 1w EMA (bearish alignment)
        elif (ema_6h[i] < ema_1d_aligned[i] < ema_1w_aligned[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit conditions: breakdown of alignment
        elif position == 1 and not (ema_6h[i] > ema_1d_aligned[i] > ema_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and not (ema_6h[i] < ema_1d_aligned[i] < ema_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals