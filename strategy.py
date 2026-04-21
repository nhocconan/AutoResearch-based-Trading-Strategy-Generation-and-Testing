#!/usr/bin/env python3
"""
4h_12h_1d_Camarilla_InsideBar_Volume_Filter_v1
Hypothesis: Breakout above H3 or below L3 on inside bar with volume spike and 12h trend filter.
Long when: inside bar + close > H3 + 12h EMA34 rising + volume spike.
Short when: inside bar + close < L3 + 12h EMA34 falling + volume spike.
Exit at H4/L4 or reversal at H3/L3.
Inside bar reduces false breakouts; volume confirms conviction; 12h EMA filters counter-trend.
Target: 20-35 trades/year per symbol.
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
    
    # Load 12h data for EMA34 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Align to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        prev_price = prices['close'].iloc[i-1]
        volume = prices['volume'].iloc[i]
        
        # Inside bar: current high <= previous high AND current low >= previous low
        high = prices['high'].iloc[i]
        low = prices['low'].iloc[i]
        prev_high = prices['high'].iloc[i-1]
        prev_low = prices['low'].iloc[i-1]
        inside_bar = (high <= prev_high) and (low >= prev_low)
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long: inside bar breakout above H3 with bullish 12h trend and volume
            if (inside_bar and 
                price > h3_aligned[i] and 
                ema_12h_aligned[i] > ema_12h_aligned[i-1] and  # rising
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short: inside bar breakout below L3 with bearish 12h trend and volume
            elif (inside_bar and 
                  price < l3_aligned[i] and 
                  ema_12h_aligned[i] < ema_12h_aligned[i-1] and  # falling
                  volume_ok):
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

name = "4h_12h_1d_Camarilla_InsideBar_Volume_Filter_v1"
timeframe = "4h"
leverage = 1.0