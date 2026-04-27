#!/usr/bin/env python3
"""
Multi-timeframe strategy combining 4h price action with 1h entry timing.
Hypothesis: 4h momentum filters reduce false signals in choppy markets,
while 1h entry timing captures momentum bursts with controlled frequency.
Trades target: 20-40/year (80-160 total over 4 years) to avoid fee drag.
Works in bull/bear via momentum + volatility filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for momentum filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h momentum (close vs open)
    close_4h = df_4h['close'].values
    open_4h = df_4h['open'].values
    momentum_4h = close_4h - open_4h
    
    # Align 4h momentum to 1h timeframe
    momentum_4h_aligned = align_htf_to_ltf(prices, df_4h, momentum_4h)
    
    # 1h volatility filter (ATR-based)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]
    
    # 1h momentum (price change)
    price_change = np.diff(close, prepend=close[0])
    
    # Volume filter
    vol_ma = np.zeros(n)
    vol_ma[0] = volume[0]
    for i in range(1, n):
        vol_ma[i] = 0.9 * vol_ma[i-1] + 0.1 * volume[i]
    
    signals = np.zeros(n)
    position = 0
    size = 0.20  # 20% position size
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
            
        # Skip if data not ready
        if np.isnan(momentum_4h_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
            
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        mom_threshold = 0.5 * atr[i]  # Dynamic threshold based on volatility
        
        if position == 0:
            # Long: 4h bullish momentum + 1h price up + volume surge
            if (momentum_4h_aligned[i] > mom_threshold and 
                price_change[i] > 0 and 
                vol_ratio > 1.5):
                signals[i] = size
                position = 1
            # Short: 4h bearish momentum + 1h price down + volume surge
            elif (momentum_4h_aligned[i] < -mom_threshold and 
                  price_change[i] < 0 and 
                  vol_ratio > 1.5):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: 4h momentum turns bearish OR price reverses
            if momentum_4h_aligned[i] < -mom_threshold * 0.5 or price_change[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: 4h momentum turns bullish OR price reverses
            if momentum_4h_aligned[i] > mom_threshold * 0.5 or price_change[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Momentum_1h_Volume_Entry"
timeframe = "1h"
leverage = 1.0