#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_VolumeATRFilter_v1
Hypothesis: Breakout at Camarilla R1/S1 levels with volume confirmation and ATR-based stop on 12h timeframe.
Uses 1w trend filter to avoid counter-trend trades. Target: 12-37 trades/year per symbol.
Works in bull/bear: In uptrend (price > 1w EMA34), buy R1 breakouts; in downtrend (price < 1w EMA34), sell S1 breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data once for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.0 / 12
    s1 = prev_close - rang * 1.0 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1w EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # Trend filter: 1w EMA34
        uptrend = price > ema_34_1w_aligned[i]
        downtrend = price < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in uptrend
            if price > r1_aligned[i] and volume_ok and uptrend:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 with volume in downtrend
            elif price < s1_aligned[i] and volume_ok and downtrend:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Stoploss: exit if price drops 2.5 * ATR from entry
            if price < entry_price - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Stoploss: exit if price rises 2.5 * ATR from entry
            if price > entry_price + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_VolumeATRFilter_v1"
timeframe = "12h"
leverage = 1.0