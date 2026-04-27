#!/usr/bin/env python3
"""
4h_OrderBlock_Bounce_Strategy
Trades institutional order block bounces using liquidity zones (equal highs/lows) and volume confirmation.
Long when price revisits a bullish order block (equal lows) with bullish engulfing candle and volume spike.
Short when price revisits a bearish order block (equal highs) with bearish engulfing candle and volume spike.
Uses 1d trend filter (EMA50) to align with higher timeframe direction.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Equal highs/lows detection (liquidity zones)
    def find_equal_levels(arr, lookback=20, tolerance=0.001):
        """Find equal highs/lows within tolerance percentage"""
        n = len(arr)
        levels = np.full(n, np.nan)
        for i in range(lookback, n):
            window = arr[i-lookback:i]
            if len(window) < 2:
                continue
            # Check for equal highs
            max_val = np.max(window)
            max_indices = np.where(np.abs(window - max_val) <= (max_val * tolerance))[0]
            if len(max_indices) >= 2:
                levels[i] = max_val
            # Check for equal lows
            min_val = np.min(window)
            min_indices = np.where(np.abs(window - min_val) <= (min_val * tolerance))[0]
            if len(min_indices) >= 2:
                if np.isnan(levels[i]):
                    levels[i] = min_val
                else:
                    # If both, prioritize based on price action
                    levels[i] = min_val if close[i] < np.mean(window) else max_val
        return levels
    
    equal_highs = find_equal_levels(high, lookback=20, tolerance=0.0015)
    equal_lows = find_equal_levels(low, lookback=20, tolerance=0.0015)
    
    # Volume spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period - 1, n):
        vol_ma[i] = np.mean(volume[i-vol_period+1:i+1])
    volume_spike = volume > (vol_ma * 2.0)  # Volume at least 2x average
    
    # Engulfing candle detection
    bullish_engulfing = (close > np.roll(open_prices, 1)) & (open_prices < np.roll(close, 1))
    bearish_engulfing = (close < np.roll(open_prices, 1)) & (open_prices > np.roll(close, 1))
    # Handle first element
    bullish_engulfing[0] = False
    bearish_engulfing[0] = False
    
    # Get 1d data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_1d_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_1d_period:
        ema_1d[ema_1d_period - 1] = np.mean(close_1d[:ema_1d_period])
        for i in range(ema_1d_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_1d_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_1d_period + 1))))
    
    # Align 1d EMA50 to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Need open prices for engulfing detection
    open_prices = prices['open'].values
    
    # Warmup: need equal levels, volume MA, and EMA1d
    start_idx = max(40, vol_period - 1, ema_1d_period - 1)  # 40 for equal levels lookback
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(equal_highs[i]) and np.isnan(equal_lows[i])) or \
           np.isnan(vol_ma[i]) or np.isnan(ema_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        eq_high = equal_highs[i]
        eq_low = equal_lows[i]
        vol_spike = volume_spike[i]
        bull_eng = bullish_engulfing[i]
        bear_eng = bearish_engulfing[i]
        ema1d_val = ema_1d_aligned[i]
        
        if position == 0:
            # Long: price at bullish order block (equal lows) + bullish engulfing + volume spike + above 1d EMA50
            if (not np.isnan(eq_low) and 
                abs(price - eq_low) <= (eq_low * 0.002) and  # Within 0.2% of equal low
                bull_eng and 
                vol_spike and 
                price > ema1d_val):
                signals[i] = size
                position = 1
            # Short: price at bearish order block (equal highs) + bearish engulfing + volume spike + below 1d EMA50
            elif (not np.isnan(eq_high) and 
                  abs(price - eq_high) <= (eq_high * 0.002) and  # Within 0.2% of equal high
                  bear_eng and 
                  vol_spike and 
                  price < ema1d_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches opposite order block or volume divergence
            if (not np.isnan(eq_high) and 
                abs(price - eq_high) <= (eq_high * 0.002)) or \
               (volume < vol_ma[i] * 0.5):  # Volume drops below 50% of average
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price reaches opposite order block or volume divergence
            if (not np.isnan(eq_low) and 
                abs(price - eq_low) <= (eq_low * 0.002)) or \
               (volume < vol_ma[i] * 0.5):  # Volume drops below 50% of average
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_OrderBlock_Bounce_Strategy"
timeframe = "4h"
leverage = 1.0