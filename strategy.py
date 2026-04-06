#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with 1-day weekly pivot (R4/S4) breakout confirmation.
# Uses weekly pivot levels from daily data (calculated from prior week) to filter breakouts:
# - Long: price breaks above Donchian(20) high AND above weekly R4 (strong bullish breakout)
# - Short: price breaks below Donchian(20) low AND below weekly S4 (strong bearish breakout)
# Weekly pivots act as institutional support/resistance; breaks indicate sustained momentum.
# Works in bull markets by catching strong continuations, in bear markets by avoiding false breakdowns.
# Target: 80-160 total trades over 4 years (20-40/year).
name = "exp_14151_6h_donchian20_1d_weekly_pivot_v1"
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

def calculate_weekly_pivots(high, low, close):
    """Calculate weekly pivot levels from daily OHLC (using prior week's data)
    R4 = R3 + (high - low)
    R3 = R2 + (high - low)
    R2 = R1 + (high - low)
    R1 = 2 * PP - low
    PP = (high + low + close) / 3
    S1 = 2 * PP - high
    S2 = S1 - (high - low)
    S3 = S2 - (high - low)
    S4 = S3 - (high - low)
    """
    # Typical price for pivot point
    pp = (high + low + close) / 3.0
    # Daily range
    rng = high - low
    
    # Weekly pivot levels (using prior week's data)
    r1 = 2 * pp - low
    s1 = 2 * pp - high
    r2 = r1 + rng
    s2 = s1 - rng
    r3 = r2 + rng
    s3 = s2 - rng
    r4 = r3 + rng
    s4 = s3 - rng
    
    return pp, r1, r2, r3, r4, s1, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for weekly pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot levels from daily data
    pp, r1, r2, r3, r4, s1, s2, s3, s4 = calculate_weekly_pivots(high_1d, low_1d, close_1d)
    
    # Align weekly pivot levels to 6h timeframe (use prior week's levels)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 20 for Donchian, 50 for pivots calculation, 14 for ATR)
    start = max(20, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(r4_aligned[i]) or \
           np.isnan(s4_aligned[i]) or np.isnan(atr[i]):
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
        
        # Donchian breakout signals with weekly pivot confirmation
        # Long: break above Donchian high AND above weekly R4 (strong bullish breakout)
        # Short: break below Donchian low AND below weekly S4 (strong bearish breakout)
        breakout_long = (close[i] > highest_high[i-1]) and (close[i] > r4_aligned[i])
        breakout_short = (close[i] < lowest_low[i-1]) and (close[i] < s4_aligned[i])
        
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
            # Exit long on stop or breakdown of Donchian low
            if close[i] <= stop_price or close[i] < lowest_low[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or breakout of Donchian high
            if close[i] >= stop_price or close[i] > highest_high[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals