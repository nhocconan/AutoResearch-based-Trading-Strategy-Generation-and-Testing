#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume_ATRFilter_v1
Hypothesis: Price breaks above/below daily Camarilla R1/S1 levels with volume confirmation and ATR-based stoploss on 12h timeframe.
In bull market, buy breakouts above R1; in bear market, sell breakdowns below S1.
Volume filter ensures breakouts have conviction. ATR stoploss limits downside.
Works in both bull and bear markets by trading breakouts with the trend.
Target: 12-37 trades/year per symbol (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for daily Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Daily Camarilla pivot levels (R1, S1)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = pivot + 1.1 * (prev_high - prev_low) / 12.0
    s1 = pivot - 1.1 * (prev_high - prev_low) / 12.0
    
    # Align daily levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: 20-period average (10 days on 12h)
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss and breakout confirmation
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
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # Breakout conditions
        breakout_up = price > r1_aligned[i]
        breakout_down = price < s1_aligned[i]
        
        if position == 0:
            # Long: breakout above R1 with volume
            if breakout_up and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 with volume
            elif breakout_down and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price drops below R1 or stoploss hit
            if price < r1_aligned[i] - 1.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price rises above S1 or stoploss hit
            if price > s1_aligned[i] + 1.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0