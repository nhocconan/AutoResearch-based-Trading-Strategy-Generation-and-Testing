#!/usr/bin/env python3
"""
6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
- Long: price breaks above 6h Donchian(20) + weekly pivot direction up + volume > 1.5x avg
- Short: price breaks below 6h Donchian(20) + weekly pivot direction down + volume > 1.5x avg
- Exit: stop loss (2*ATR) or reversal signal
- Position size: 0.25 (25%)
- Target: 75-200 trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14227_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def calculate_atr(high, low, close, period):
    """Calculate ATR with proper min_periods"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for pivot calculation (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot levels (standard floor trader pivots)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    r4_1w = r3_1w + (high_1w - low_1w)
    s4_1w = s3_1w - (high_1w - low_1w)
    
    # Weekly pivot direction: price above/below pivot
    pivot_dir_1w = np.where(close_1w > pivot_1w, 1, -1)  # 1 = bullish, -1 = bearish
    
    # Align weekly data to 6h
    pivot_dir_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_dir_1w)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (1.5 * vol_ma)
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 20 for volume, 14 for ATR)
    start = max(20, 20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(pivot_dir_1w_aligned[i]) or \
           np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Donchian breakout signals with weekly pivot direction and volume
        # Long: break above upper band + weekly pivot bullish + volume
        # Short: break below lower band + weekly pivot bearish + volume
        breakout_long = (close[i] > highest_high[i-1]) and (pivot_dir_1w_aligned[i] == 1) and vol_filter[i]
        breakout_short = (close[i] < lowest_low[i-1]) and (pivot_dir_1w_aligned[i] == -1) and vol_filter[i]
        
        # Generate signals
        if position == 0:
            if breakout_long:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif breakout_short:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or breakdown of lower band
            if close[i] <= stop_price or close[i] < lowest_low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or breakout of upper band
            if close[i] >= stop_price or close[i] > highest_high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals