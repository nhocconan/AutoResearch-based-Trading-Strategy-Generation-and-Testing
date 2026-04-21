#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_Regime_Volume_ATRFilter_V2
Hypothesis: Donchian(20) breakout on 4h timeframe with volume confirmation and choppiness regime filter.
Long when price breaks above upper Donchian(20) in trending market (CHOP < 38.2) with volume spike.
Short when price breaks below lower Donchian(20) in trending market (CHOP < 38.2) with volume spike.
Uses ATR(20) for dynamic stoploss (signal -> 0 when adverse move > 2*ATR).
Works in bull/bear: captures strong trends in both directions, avoids whipsaw in ranging markets via chop filter.
Target: 20-50 trades/year per symbol (80-200 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_roll = prices['high'].rolling(window=20, min_periods=20).max()
    low_roll = prices['low'].rolling(window=20, min_periods=20).min()
    upper = high_roll.values
    lower = low_roll.values
    
    # Calculate ATR(20) for stoploss and volatility filter
    tr1 = prices['high'] - prices['low']
    tr2 = abs(prices['high'] - prices['close'].shift(1))
    tr3 = abs(prices['low'] - prices['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index (CHOP) for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(HH(14) - LL(14)))) / log10(14)
    atr_14 = tr.rolling(window=14, min_periods=14).sum().values
    hh_14 = prices['high'].rolling(window=14, min_periods=14).max().values
    ll_14 = prices['low'].rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr_14 / (hh_14 - ll_14)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((hh_14 - ll_14) > 0, chop_raw, 50.0)  # default to 50 (neutral) when range is zero
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    if n >= 20:
        vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
        volume_ok = prices['volume'].values > 1.8 * vol_ma
    else:
        volume_ok = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(atr[i]) or np.isnan(chop[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        close_price = prices['close'].iloc[i]
        
        # Regime filter: only trade in trending markets (CHOP < 38.2)
        is_trending = chop[i] < 38.2
        
        if position == 0:
            # Long entry: price breaks above upper Donchian + volume + trending market
            if close_price > upper[i] and volume_ok[i] and is_trending:
                signals[i] = 0.25
                position = 1
                entry_price = close_price
            # Short entry: price breaks below lower Donchian + volume + trending market
            elif close_price < lower[i] and volume_ok[i] and is_trending:
                signals[i] = -0.25
                position = -1
                entry_price = close_price
        
        elif position == 1:
            # Long exit: stoploss (2*ATR adverse move) or Donchian mean reversion (back below midpoint)
            if close_price <= entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stoploss (2*ATR adverse move) or Donchian mean reversion (back above midpoint)
            if close_price >= entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_Regime_Volume_ATRFilter_V2"
timeframe = "4h"
leverage = 1.0