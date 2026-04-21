#!/usr/bin/env python3
"""
1h_4h_1d_Camarilla_Breakout_Volume_Momentum_v1
Hypothesis: Breakout above/below H3/L3 on 1h with 4h trend confirmation (above/below EMA34) and volume spike.
Long when price breaks above H3 with 4h EMA34 up and volume spike.
Short when price breaks below L3 with 4h EMA34 down and volume spike.
Exit when price returns to H3/L3 or reaches H4/L4.
Works in bull by following 4h trend for breakouts, in bear by shorting breakdowns with trend.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate daily levels for Camarilla
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
    
    # Align to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # Calculate 4h EMA34 for trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.8 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.8 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: break above H3 with bullish trend and volume
            if (price > h3_aligned[i] and 
                ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1] and  # 4h EMA rising
                volume_ok):
                signals[i] = 0.20
                position = 1
            # Short conditions: break below L3 with bearish trend and volume
            elif (price < l3_aligned[i] and 
                  ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1] and  # 4h EMA falling
                  volume_ok):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: return to H3 or reach H4
            if price <= h3_aligned[i] or price >= h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: return to L3 or reach L4
            if price >= l3_aligned[i] or price <= l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_Camarilla_Breakout_Volume_Momentum_v1"
timeframe = "1h"
leverage = 1.0