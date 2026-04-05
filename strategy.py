#!/usr/bin/env python3
"""
Experiment #9555: 6h 1-Week Momentum + Daily Volume Filter.
Hypothesis: On 6h timeframe, price above 1-week high for 2+ consecutive bars with volume above 
20-period moving average indicates strong momentum continuation. This works in both bull and bear 
markets because it captures trending moves regardless of direction. Targets 80-160 total trades 
over 4 years (20-40/year) by requiring both momentum and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_9555_6h_1w_momentum_volume_filter"
timeframe = "6h"
leverage = 1.0

# Parameters
WEEK_HIGH_PERIOD = 5  # 5 days = 1 week (for 1d data)
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 2.0
MIN_CONSECUTIVE_BARS = 2
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for 1-week high calculation)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-week high (highest high over past 5 days)
    high_1d = df_1d['high'].values
    week_high = pd.Series(high_1d).rolling(window=WEEK_HIGH_PERIOD, min_periods=WEEK_HIGH_PERIOD).max().values
    
    # Align 1-week high to 6h timeframe
    week_high_aligned = align_htf_to_ltf(prices, df_1d, week_high)
    
    # Calculate LTF indicators (6h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume moving average for filter
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(WEEK_HIGH_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + MIN_CONSECUTIVE_BARS
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(week_high_aligned[i]) or np.isnan(volume_ma[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Momentum condition: price above 1-week high for MIN_CONSECUTIVE_BARS consecutive bars
        if i >= MIN_CONSECUTIVE_BARS:
            momentum_up = all(close[i-j] > week_high_aligned[i-j] for j in range(MIN_CONSECUTIVE_BARS))
            momentum_down = all(close[i-j] < week_high_aligned[i-j] for j in range(MIN_CONSECUTIVE_BARS))
        else:
            momentum_up = False
            momentum_down = False
        
        # Volume filter: current volume above threshold * moving average
        volume_filter = volume[i] > (VOLUME_THRESHOLD * volume_ma[i]) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = momentum_up and volume_filter
        short_entry = momentum_down and volume_filter
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals