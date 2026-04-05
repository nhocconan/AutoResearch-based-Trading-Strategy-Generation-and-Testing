#!/usr/bin/env python3
"""
exp_7515_6h_1w_1d_donchian_vol_v1
Hypothesis: 6s Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Weekly pivot defines long-term trend: price above weekly pivot = bullish bias (long breakouts),
price below weekly pivot = bearish bias (short breakdowns). Volume > 1.5x 20-period average
confirms breakout strength. Targets 50-150 trades over 4 years (12-37/year) with strict
breakout conditions + trend alignment. Works in bull/bear by following weekly trend.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7515_6h_1w_1d_donchian_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_AVG_PERIOD = 20
VOLUME_THRESHOLD = 1.5  # 1.5x average volume
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Support/resistance levels
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    
    # Align weekly pivot to 6s timeframe (use pivot for trend bias)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    
    # Calculate 1d average volume for confirmation
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=VOLUME_AVG_PERIOD, min_periods=VOLUME_AVG_PERIOD).mean().values
    avg_volume_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_volume_1d)
    
    # Calculate LTF indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOLUME_AVG_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(pivot_1w_aligned[i]) or np.isnan(avg_volume_1d_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine weekly trend bias
        bullish_bias = close[i] > pivot_1w_aligned[i]   # price above weekly pivot
        bearish_bias = close[i] < pivot_1w_aligned[i]   # price below weekly pivot
        
        # Volume confirmation
        volume_confirmed = volume[i] > (VOLUME_THRESHOLD * avg_volume_1d_aligned[i])
        
        # Breakout conditions
        bullish_breakout = (high[i] > highest_high[i]) and volume_confirmed
        bearish_breakout = (low[i] < lowest_low[i]) and volume_confirmed
        
        # Entry conditions: follow weekly trend bias
        long_entry = bullish_bias and bullish_breakout
        short_entry = bearish_bias and bearish_breakout
        
        # Exit conditions: opposite breakout or reversal
        long_exit = bearish_breakout  # exit long on bearish breakout
        short_exit = bullish_breakout  # exit short on bullish breakout
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals