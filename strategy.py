#!/usr/bin/env python3
"""
4h_Donchian20_VolumeRegime
Hypothesis: 4-hour price breakout beyond 20-period Donchian channels with volume confirmation and 1-week EMA34 trend filter captures institutional momentum while avoiding whipsaw. Works in bull/bear markets by filtering breakouts with higher timeframe trend. Target 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load weekly data once for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema34_1w = np.zeros_like(close_1w)
    ema34_1w[0] = close_1w[0]
    alpha = 2.0 / (34 + 1)
    for i in range(1, len(close_1w)):
        ema34_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema34_1w[i-1]
    
    # Align weekly EMA34 to 4h timeframe
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    
    for i in range(n):
        if i < 19:
            upper[i] = np.max(high[0:i+1]) if i >= 0 else high[i]
            lower[i] = np.min(low[0:i+1]) if i >= 0 else low[i]
        else:
            upper[i] = np.max(high[i-19:i+1])
            lower[i] = np.min(low[i-19:i+1])
    
    # Volume filter: volume > 1.3x 20-period average
    volume = prices['volume'].values
    volume_avg = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            volume_avg[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            volume_avg[i] = np.mean(volume[i-19:i+1])
    volume_filter = volume > (1.3 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema34_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        up = upper[i]
        low_ch = lower[i]
        ema34 = ema34_1w_aligned[i]
        vol_confirm = volume_filter[i]
        
        # Calculate ATR for stoploss (20-period)
        if i >= 20:
            tr_values = []
            for j in range(1, 21):
                idx = i - j
                if idx >= 0:
                    tr = max(high[idx] - low[idx], 
                             abs(high[idx] - close[idx-1]), 
                             abs(low[idx] - close[idx-1]))
                    tr_values.append(tr)
            atr = np.mean(tr_values) if tr_values else 0
        else:
            atr = 0
        
        # Stoploss: 2.0 * ATR from entry
        if position == 1 and price < entry_price - 2.0 * atr:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and price > entry_price + 2.0 * atr:
            signals[i] = 0.0
            position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume confirmation in uptrend (price > weekly EMA34)
            if price > up and vol_confirm and price > ema34:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian with volume confirmation in downtrend (price < weekly EMA34)
            elif price < low_ch and vol_confirm and price < ema34:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price returns to midpoint or trend breaks
            midpoint = (up + low_ch) / 2
            if price < midpoint or price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to midpoint or trend breaks
            midpoint = (up + low_ch) / 2
            if price > midpoint or price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeRegime"
timeframe = "4h"
leverage = 1.0