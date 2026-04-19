#!/usr/bin/env python3
"""
6h_Wyckoff_Spring_Upthrust_V1
Hypothesis: Wyckoff accumulation/distribution patterns on 6h timeframe.
- Spring: False breakdown below support followed by quick reversal (accumulation)
- Upthrust: False breakout above resistance followed by quick reversal (distribution)
- Works in both bull/bear by identifying smart money traps before reversals
- Uses price action and volume spread analysis (VSA) for confirmation
- Targets 50-150 total trades over 4 years (12-37/year) with selective entries
"""

name = "6h_Wyckoff_Spring_Upthrust_V1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume analysis: compare current volume to recent average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = np.divide(volume, volume_ma, out=np.ones_like(volume), where=volume_ma!=0)
    
    # Identify potential support/resistance levels using swing points
    def find_swing_points(high, low, lookback=10):
        """Find swing highs and lows"""
        swing_high = np.full_like(high, np.nan)
        swing_low = np.full_like(low, np.nan)
        
        for i in range(lookback, len(high) - lookback):
            # Swing high: higher than lookback periods before and after
            if high[i] == np.max(high[i-lookback:i+lookback+1]):
                swing_high[i] = high[i]
            # Swing low: lower than lookback periods before and after
            if low[i] == np.min(low[i-lookback:i+lookback+1]):
                swing_low[i] = low[i]
        
        return swing_high, swing_low
    
    swing_high, swing_low = find_swing_points(high, low, lookback=5)
    
    # Get 1d data for higher context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d ATR for volatility normalization
    def calculate_atr(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        atr = np.full_like(close, np.nan)
        if len(tr) >= period:
            atr[period-1] = np.mean(tr[:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_1d = calculate_atr(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, 14)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if required data is unavailable
        if np.isnan(atr_1d_aligned[i]) or atr_1d_aligned[i] == 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Normalized price ranges for the last 10 periods
        lookback = 10
        if i < lookback:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        period_low = np.min(low[i-lookback:i+1])
        period_high = np.max(high[i-lookback:i+1])
        period_range = period_high - period_low
        
        if period_range == 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Calculate where current price sits in the recent range
        price_position = (close[i] - period_low) / period_range  # 0 = bottom, 1 = top
        
        # Volume condition: above average volume suggests participation
        strong_volume = volume_ratio[i] > 1.2
        
        # Spring pattern: price tests recent low, closes back above it on strong volume
        # Indicates absorption of selling pressure (accumulation)
        is_spring = (
            low[i] <= period_low * 1.005 and  # Tests or slightly breaks low
            close[i] > period_low and         # Closes back above the low
            price_position > 0.3 and          # Not at extreme bottom
            strong_volume                     # Confirmed by volume
        )
        
        # Upthrust pattern: price tests recent high, closes back below it on strong volume
        # Indicates lack of buying interest (distribution)
        is_upthrust = (
            high[i] >= period_high * 0.995 and  # Tests or slightly breaks high
            close[i] < period_high and          # Closes back below the high
            price_position < 0.7 and            # Not at extreme top
            strong_volume                       # Confirmed by volume
        )
        
        # Additional context: avoid trading against strong momentum
        # Calculate short-term momentum
        if i >= 5:
            momentum = (close[i] - close[i-5]) / close[i-5]
            weak_momentum = abs(momentum) < 0.02  # Avoid strong trending markets
        else:
            weak_momentum = True
        
        if position == 0:
            # Long on spring (false breakdown reversed)
            if is_spring and weak_momentum:
                signals[i] = 0.25
                position = 1
            # Short on upthrust (false breakout reversed)
            elif is_upthrust and weak_momentum:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long exit: price shows weakness or reaches target area
            exit_condition = (
                price_position > 0.8 or  # Reached upper part of range
                (close[i] < close[i-1] and volume_ratio[i] > 1.5) or  # Weak close on high volume
                low[i] < period_low  # Broke below the spring low
            )
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short exit: price shows strength or reaches target area
            exit_condition = (
                price_position < 0.2 or  # Reached lower part of range
                (close[i] > close[i-1] and volume_ratio[i] > 1.5) or  # Strong close on high volume
                high[i] > period_high  # Broke above the upthrust high
            )
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals