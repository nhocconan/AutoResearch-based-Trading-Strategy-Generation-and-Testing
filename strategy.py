#!/usr/bin/env python3
"""
Experiment #10399: 6h Chandelier Exit Reversal with 12h Trend Filter
Hypothesis: Chandelier Exit (trailing stop based on ATR) provides dynamic support/resistance.
In ranging markets, price reverses at these levels; in trending markets, trailing stop follows.
Combined with 12h EMA trend filter to only take reversals in direction of higher timeframe trend.
Works in both bull/bear: reversals catch pullbacks in uptrend and bounces in downtrend.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_10399_6h_chandelier_exit_reversal_12h_trend_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CHANDELIER_PERIOD = 22
ATR_PERIOD = 22
CHANDELIER_MULTIPLIER = 3.0
TREND_EMA_PERIOD = 50
SIGNAL_SIZE = 0.25

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_chandelier_exit(high, low, close, atr, period, multiplier):
    """Calculate Chandelier Exit (long and short)"""
    # For long positions: highest high - ATR * multiplier
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    chandelier_long = highest_high - (atr * multiplier)
    
    # For short positions: lowest low + ATR * multiplier
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    chandelier_short = lowest_low + (atr * multiplier)
    
    return chandelier_long, chandelier_short

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate 12h EMA for trend direction
    close_12h = df_12h['close'].values
    ema_12h = calculate_ema(close_12h, TREND_EMA_PERIOD)
    
    # Align 12h EMA to 6h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # ATR for Chandelier Exit
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Chandelier Exit levels
    chandelier_long, chandelier_short = calculate_chandelier_exit(
        high, low, close, atr, CHANDELIER_PERIOD, CHANDELIER_MULTIPLIER
    )
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from warmup period
    start = max(CHANDELIER_PERIOD, ATR_PERIOD, TREND_EMA_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if 12h EMA not available
        if np.isnan(ema_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 12h EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Chandelier Exit reversal signals
        # Long: price crosses above Chandelier long (short covering)
        long_signal = (close[i] > chandelier_long[i]) and (close[i-1] <= chandelier_long[i-1]) if not np.isnan(chandelier_long[i]) else False
        # Short: price crosses below Chandelier short (long liquidation)
        short_signal = (close[i] < chandelier_short[i]) and (close[i-1] >= chandelier_short[i-1]) if not np.isnan(chandelier_short[i]) else False
        
        # Generate signals
        if position == 0:
            # Only take longs in uptrend, shorts in downtrend
            if long_signal and uptrend:
                signals[i] = SIGNAL_SIZE
                position = 1
            elif short_signal and downtrend:
                signals[i] = -SIGNAL_SIZE
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: exit when price crosses below Chandelier long
            if close[i] < chandelier_long[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            # Short: exit when price crosses above Chandelier short
            if close[i] > chandelier_short[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals