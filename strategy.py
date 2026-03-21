#!/usr/bin/env python3
"""
EXPERIMENT #011 - HMA Crossover with Daily Trend Filter (12h)
=============================================================
Hypothesis: 12h HMA crossover provides cleaner signals than 4h/1h while maintaining
responsiveness. Adding 1d HMA trend filter ensures we only trade in direction of 
major trend, reducing whipsaws. ATR-based stoploss protects against adverse moves.

Key features:
- Primary TF: 12h (new for this experiment)
- HTF filter: 1d HMA(21) for trend direction
- Entry: HMA(8) crosses HMA(21) on 12h
- Stoploss: 2*ATR(14) trailing
- Position sizing: 0.25-0.30 discrete levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "hma_crossover_daily_filter_12h_v1"
timeframe = "12h"
leverage = 1.0


def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    close_s = pd.Series(close)
    wma1 = close_s.ewm(span=period//2, adjust=False).mean()
    wma2 = close_s.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma1 - wma2
    hma = raw_hma.ewm(span=int(np.sqrt(period)), adjust=False).mean()
    return hma.values


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1d HTF data ONCE before loop (Rule 1)
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # Calculate 12h indicators
    hma_fast = calculate_hma(close, 8)
    hma_slow = calculate_hma(close, 21)
    
    # Calculate ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.25  # Base position size
    MAX_SIZE = 0.30
    
    # Track position state for stoploss
    entry_price = 0.0
    position_side = 0  # 0=flat, 1=long, -1=short
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    min_period = 21  # Wait for indicators to stabilize
    
    for i in range(min_period, n):
        # Check for NaN in any indicator
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]) or np.isnan(atr[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Daily trend filter
        daily_trend = 1 if close[i] > hma_1d_aligned[i] else -1
        
        # HMA crossover signal
        crossover_signal = 0
        if hma_fast[i] > hma_slow[i] and hma_fast[i-1] <= hma_slow[i-1]:
            crossover_signal = 1  # Bullish crossover
        elif hma_fast[i] < hma_slow[i] and hma_fast[i-1] >= hma_slow[i-1]:
            crossover_signal = -1  # Bearish crossover
        
        # Determine target signal based on trend filter
        target_signal = 0.0
        if crossover_signal == daily_trend and crossover_signal != 0:
            target_signal = SIZE * crossover_signal
        
        # Stoploss logic (Rule 6) - check BEFORE setting new signal
        stoploss_triggered = False
        if position_side != 0:
            if position_side == 1:
                # Long position - update highest
                highest_since_entry = max(highest_since_entry, close[i])
                trailing_stop = highest_since_entry - 2.0 * atr[i]
                if close[i] < trailing_stop:
                    stoploss_triggered = True
            else:
                # Short position - update lowest
                lowest_since_entry = min(lowest_since_entry, close[i])
                trailing_stop = lowest_since_entry + 2.0 * atr[i]
                if close[i] > trailing_stop:
                    stoploss_triggered = True
        
        if stoploss_triggered:
            signals[i] = 0.0
            position_side = 0
            entry_price = 0.0
            highest_since_entry = 0.0
            lowest_since_entry = float('inf')
        else:
            # Apply signal change
            if target_signal != 0.0:
                signals[i] = target_signal
                if position_side == 0:
                    # New entry
                    position_side = 1 if target_signal > 0 else -1
                    entry_price = close[i]
                    highest_since_entry = close[i]
                    lowest_since_entry = close[i]
            elif position_side != 0:
                # Maintain existing position
                signals[i] = SIZE * position_side
            else:
                signals[i] = 0.0
    
    return signals