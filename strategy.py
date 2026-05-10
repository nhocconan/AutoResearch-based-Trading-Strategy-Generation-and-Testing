#!/usr/bin/env python3
"""
6h_ADX_Supertrend_Slope
Hypothesis: Use ADX(14) on 6h to identify trending regimes (ADX>25) and Supertrend(ATR=10, mult=3) for direction.
Enter long when Supertrend turns bullish in strong trend, short when bearish. Exit when ADX weakens or Supertrend reverses.
Adds 6h-specific momentum filter: require price to be above/below EMA20 for entry.
Works in bull/bear by only taking trades in strong trend regimes, avoiding sideways chop.
Target: 15-25 trades/year (60-100 total) to minimize fee drag.
"""

name = "6h_ADX_Supertrend_Slope"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def true_range(high, low, close):
    """Calculate True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    return np.maximum(tr1, np.maximum(tr2, tr3))

def atr(high, low, close, period):
    """Calculate Average True Range"""
    tr = true_range(high, low, close)
    atr_vals = np.full_like(tr, np.nan, dtype=float)
    if len(tr) >= period:
        atr_vals[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr_vals[i] = (atr_vals[i-1] * (period-1) + tr[i]) / period
    return atr_vals

def supertrend(high, low, close, atr_period, multiplier):
    """Calculate Supertrend indicator"""
    atr_vals = atr(high, low, close, atr_period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr_vals
    lower_band = hl2 - multiplier * atr_vals
    
    supertrend = np.full_like(close, np.nan, dtype=float)
    direction = np.full_like(close, np.nan, dtype=float)  # 1 for up, -1 for down
    
    for i in range(len(close)):
        if np.isnan(atr_vals[i]) or i == 0:
            continue
        if i == atr_period:
            # Initialize first valid supertrend
            if close[i] > upper_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1
        else:
            # Calculate upper and lower bands
            if supertrend[i-1] == upper_band[i-1]:
                upper_band[i] = min(upper_band[i], upper_band[i-1])
            else:
                upper_band[i] = max(upper_band[i], upper_band[i-1])
                
            if supertrend[i-1] == lower_band[i-1]:
                lower_band[i] = max(lower_band[i], lower_band[i-1])
            else:
                lower_band[i] = min(lower_band[i], lower_band[i-1])
            
            # Determine trend
            if close[i] > upper_band[i]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            elif close[i] < lower_band[i]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            elif direction[i-1] == -1 and close[i] > supertrend[i-1]:
                direction[i] = 1
                supertrend[i] = lower_band[i]
            elif direction[i-1] == 1 and close[i] < supertrend[i-1]:
                direction[i] = -1
                supertrend[i] = upper_band[i]
            else:
                direction[i] = direction[i-1]
                supertrend[i] = supertrend[i-1]
    
    return supertrend, direction

def adx(high, low, close, period):
    """Calculate Average Directional Index"""
    # Calculate +DM and -DM
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Calculate TR
    tr = true_range(high, low, close)
    
    # Smooth using Wilder's smoothing (similar to EMA but different alpha)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr_vals = wilders_smoothing(tr, period)
    plus_di = 100 * wilders_smoothing(plus_dm, period) / atr_vals
    minus_di = 100 * wilders_smoothing(minus_dm, period) / atr_vals
    
    # Calculate DX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = np.where((plus_di + minus_di) == 0, 0, dx)
    
    # Calculate ADX
    adx_vals = wilders_smoothing(dx, period)
    return adx_vals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Supertrend and ADX on 6h data
    st, st_dir = supertrend(high, low, close, 10, 3)
    adx_vals = adx(high, low, close, 14)
    
    # EMA20 for momentum filter
    ema20 = np.full_like(close, np.nan, dtype=float)
    if len(close) >= 20:
        ema20[19] = np.mean(close[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close)):
            ema20[i] = alpha * close[i] + (1 - alpha) * ema20[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Wait for ADX and EMA20
    
    for i in range(start_idx, n):
        if np.isnan(st[i]) or np.isnan(adx_vals[i]) or np.isnan(ema20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_vals[i] > 25
        
        # Supertrend direction
        st_bullish = st_dir[i] == 1
        st_bearish = st_dir[i] == -1
        
        # Price relative to EMA20 for momentum confirmation
        price_above_ema20 = close[i] > ema20[i]
        price_below_ema20 = close[i] < ema20[i]
        
        if position == 0:
            # Enter long: Supertrend bullish, strong trend, price above EMA20
            if st_bullish and strong_trend and price_above_ema20:
                signals[i] = 0.25
                position = 1
            # Enter short: Supertrend bearish, strong trend, price below EMA20
            elif st_bearish and strong_trend and price_below_ema20:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Supertrend turns bearish OR trend weakens
            if not st_bullish or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Supertrend turns bullish OR trend weakens
            if not st_bearish or not strong_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals