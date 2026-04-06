#!/usr/bin/env python3
"""
6h ADX + Williams Alligator combination with 1d trend filter
Hypothesis: ADX filters trending regimes while Alligator identifies entry points.
Uses 1d EMA50 for trend bias to work in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_adx_alligator_1d_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # ADX components
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        down = low[i-1] - low[i]
        plus_dm[i] = up if up > down and up > 0 else 0
        minus_dm[i] = down if down > up and down > 0 else 0
    
    # Smoothed values
    tr14 = np.zeros(n)
    plus_dm14 = np.zeros(n)
    minus_dm14 = np.zeros(n)
    
    if n >= 14:
        tr14[13] = np.sum(tr[1:14])
        plus_dm14[13] = np.sum(plus_dm[1:14])
        minus_dm14[13] = np.sum(minus_dm[1:14])
        
        for i in range(14, n):
            tr14[i] = tr14[i-1] - (tr14[i-1] / 14) + tr[i]
            plus_dm14[i] = plus_dm14[i-1] - (plus_dm14[i-1] / 14) + plus_dm[i]
            minus_dm14[i] = minus_dm14[i-1] - (minus_dm14[i-1] / 14) + minus_dm[i]
    
    # DI and DX
    plus_di = np.full(n, np.nan)
    minus_di = np.full(n, np.nan)
    dx = np.full(n, np.nan)
    
    for i in range(14, n):
        if tr14[i] != 0:
            plus_di[i] = 100 * plus_dm14[i] / tr14[i]
            minus_di[i] = 100 * minus_dm14[i] / tr14[i]
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = 100 * np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
    
    # ADX (smoothed DX)
    adx = np.full(n, np.nan)
    if n >= 28:  # Need 14 for DX + 14 for smoothing
        adx[27] = np.nanmean(dx[14:28])
        for i in range(28, n):
            adx[i] = (dx[i] * 13 + adx[i-1]) / 14
    
    # Williams Alligator (13,8,5 SMAs with future shift)
    jaw = np.full(n, np.nan)   # 13-period
    teeth = np.full(n, np.nan) # 8-period
    lips = np.full(n, np.nan)  # 5-period
    
    if n >= 13:
        for i in range(13, n):
            jaw[i] = np.mean(close[i-12:i+1])  # Includes current bar
    
    if n >= 8:
        for i in range(8, n):
            teeth[i] = np.mean(close[i-7:i+1])
    
    if n >= 5:
        for i in range(5, n):
            lips[i] = np.mean(close[i-4:i+1])
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA50 on 1d close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 48) / 50
    
    # 1d trend: above EMA50 = bullish, below = bearish
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 35  # Need enough data for all indicators
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(adx[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(trend_1d_aligned[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Alligator conditions: jaws (slow), teeth (medium), lips (fast)
        # Alligator sleeping: all lines intertwined
        # Alligator awake: lips > teeth > jaws (bullish) or lips < teeth < jaws (bearish)
        alligator_bullish = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_bearish = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Alligator turns bearish OR ADX weakens
            # Stoploss: price drops 2*ATR below entry
            if (not alligator_bullish or
                adx[i] < 20 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: Alligator turns bullish OR ADX weakens
            # Stoploss: price rises 2*ATR above entry
            if (not alligator_bearish or
                adx[i] < 20 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 6 bars flat
            if bars_since_entry >= 6:
                # ADX filter: only trade when trending (ADX > 25)
                strong_trend = adx[i] > 25
                
                # Long: Alligator bullish + strong trend + bullish 1d trend
                if alligator_bullish and strong_trend and trend_1d_aligned[i] == 1:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: Alligator bearish + strong trend + bearish 1d trend
                elif alligator_bearish and strong_trend and trend_1d_aligned[i] == -1:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals