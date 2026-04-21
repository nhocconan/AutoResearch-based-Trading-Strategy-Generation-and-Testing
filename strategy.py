#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_v1
Hypothesis: Breakout of 20-period Donchian channel with volume confirmation and 4h EMA trend filter.
Long when price breaks above upper band with volume spike and EMA50 > EMA200.
Short when price breaks below lower band with volume spike and EMA50 < EMA200.
Exit when price crosses the EMA50 or reaches opposite band.
Designed to work in both bull/bear by following trend filter and using volatility-based stops.
Target: 20-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Calculate EMA50 and EMA200 for trend filter
    close = prices['close'].values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(ema50[i]) or np.isnan(ema200[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long conditions: break above upper band with volume and uptrend
            if price > upper[i] and vol_ok and ema50[i] > ema200[i]:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower band with volume and downtrend
            elif price < lower[i] and vol_ok and ema50[i] < ema200[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: cross below EMA50 or reach lower band
            if price < ema50[i] or price < lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: cross above EMA50 or reach upper band
            if price > ema50[i] or price > upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_v1"
timeframe = "4h"
leverage = 1.0