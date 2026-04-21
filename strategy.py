#!/usr/bin/env python3
"""
4h_HTF_Donchian_Breakout_Volume_ATRFilter_V1
Hypothesis: Donchian(20) breakout on 4h with 1d trend filter (price > 1d EMA50) and volume confirmation (>1.5x 20-bar MA) works in both bull and bear markets. Uses ATR-based stoploss. Target: 20-50 trades/year per symbol (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels on 4h (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation (>1.5x average)
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend = price > ema50_1d_aligned[i]
        downtrend = price < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high in uptrend with volume
            if price > donch_high[i] and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low in downtrend with volume
            elif price < donch_low[i] and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below Donchian low or stoploss
            if price < donch_low[i] or price < prices['close'].iloc[i-1] - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above Donchian high or stoploss
            if price > donch_high[i] or price > prices['close'].iloc[i-1] + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_HTF_Donchian_Breakout_Volume_ATRFilter_V1"
timeframe = "4h"
leverage = 1.0