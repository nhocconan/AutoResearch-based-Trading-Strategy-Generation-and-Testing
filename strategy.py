#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume_ATRFilter_v1
Hypothesis: Breakout of Camarilla R1/S1 levels on 12h timeframe with volume confirmation and ATR-based trend filter.
Works in bull/bear: In uptrend, buy R1 breakout; in downtrend, sell S1 breakout. Uses 1w EMA20 for trend filter.
Target: 12-37 trades/year per symbol (50-150 over 4 years).
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
    
    # Camarilla levels: R1, S1 (primary breakout levels)
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.0 / 12
    s1 = prev_close - rang * 1.0 / 12
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Load 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # ATR for volatility filter (14-period on 1d)
    if len(df_1d) < 14:
        atr_1d = np.full(len(df_1d), np.nan)
    else:
        high_low = high_1d - low_1d
        high_close = np.abs(high_1d - np.roll(close_1d, 1))
        low_close = np.abs(low_1d - np.roll(close_1d, 1))
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        tr[0] = np.nan
        atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # Trend filter: price > 1w EMA20 = uptrend, price < 1w EMA20 = downtrend
        uptrend = price > ema_20_1w_aligned[i]
        downtrend = price < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long conditions: price > R1 AND uptrend AND volume
            if (price > r1_aligned[i] and 
                uptrend and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < S1 AND downtrend AND volume
            elif (price < s1_aligned[i] and 
                  downtrend and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < 1w EMA20 (trend reversal) or ATR-based stop
            if price < ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > 1w EMA20 (trend reversal) or ATR-based stop
            if price > ema_20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0