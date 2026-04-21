#!/usr/bin/env python3
"""
4h_12h_1d_Camarilla_Pullback_Volume_Momentum_v2
Hypothesis: Pullback to Camarilla H3/L3 levels with 12h momentum confirmation and volume filter.
Long when price pulls back to H3 with 12h RSI > 50 and volume spike.
Short when price pulls back to L3 with 12h RSI < 50 and volume spike.
Exit when price reaches H4/L4 or reverses at H3/L3.
Improved with proper position sizing and reduced trade frequency via stricter volume filter.
Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: H3, L3, H4, L4
    rang = prev_high - prev_low
    h3 = prev_close + 1.1 * rang / 4
    l3 = prev_close - 1.1 * rang / 4
    h4 = prev_close + 1.1 * rang / 2
    l4 = prev_close - 1.1 * rang / 2
    
    # Align to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Load 12h data for momentum (RSI)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate RSI(14) on 12h
    delta = np.diff(close_12h, prepend=close_12h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h = rsi_12h.values
    # Align to 4h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (stricter)
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: pullback to H3 with bullish momentum and volume
            if (abs(price - h3_aligned[i]) < 0.001 * h3_aligned[i] and  # near H3
                rsi_12h_aligned[i] > 50 and volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: pullback to L3 with bearish momentum and volume
            elif (abs(price - l3_aligned[i]) < 0.001 * l3_aligned[i] and  # near L3
                  rsi_12h_aligned[i] < 50 and volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: reach H4 or reverse at H3
            if price >= h4_aligned[i] or price <= h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: reach L4 or reverse at L3
            if price <= l4_aligned[i] or price >= l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_12h_1d_Camarilla_Pullback_Volume_Momentum_v2"
timeframe = "4h"
leverage = 1.0