#!/usr/bin/env python3
"""
6h 3-Bar Reversal with Volume and 1d Trend Filter
Hypothesis: 3-bar reversal patterns (bullish/bearish engulfing-like) capture short-term reversals.
Combined with 1d EMA trend filter to only take counter-trend reversals in higher timeframe trend,
and volume confirmation to ensure follow-through. Designed for moderate trade frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_3bar_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 6h data
    open_price = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 3-bar reversal patterns
    # Bullish: current bar closes above midpoint of previous bar AND previous bar closed below its open
    # Bearish: current bar closes below midpoint of previous bar AND previous bar closed above its open
    bull_engulfing = (close > (open_price + high) / 2) & (close < open_price) & (np.roll(close, 1) < np.roll(open_price, 1))
    bear_engulfing = (close < (open_price + low) / 2) & (close > open_price) & (np.roll(close, 1) > np.roll(open_price, 1))
    
    # Shift to avoid look-ahead (pattern confirmed at close of current bar)
    bull_engulfing = np.roll(bull_engulfing, 1)
    bear_engulfing = np.roll(bear_engulfing, 1)
    bull_engulfing[0] = False
    bear_engulfing[0] = False
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine 1d trend: above EMA50 = uptrend, below = downtrend
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Check exits: opposite 3-bar pattern
        if position == 1:  # long position
            # Exit: bearish 3-bar reversal
            if bear_engulfing[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: bullish 3-bar reversal
            if bull_engulfing[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: 3-bar reversal AGAINST 1d trend + volume
            # In uptrend, look for bearish reversal (short)
            # In downtrend, look for bullish reversal (long)
            vol_filter = volume[i] > vol_ma[i] * 1.5
            
            if downtrend and bull_engulfing[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            elif uptrend and bear_engulfing[i] and vol_filter:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals