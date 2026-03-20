#!/usr/bin/env python3
"""
EXPERIMENT #005 - Donchian Channel Breakout (4h) Trend Following Strategy
==========================================================================
Hypothesis: Donchian(20) breakout on 4h timeframe will capture sustained trends
with clearer entry/exit signals than Supertrend. The channel breakout method
was proven by Turtle Traders and works well in crypto's trending markets.

Key differences from Supertrend:
- Entry on confirmed breakouts (20-period high/low) vs ATR-based stops
- Exit on channel midpoint cross or opposite breakout
- Fewer false signals in ranging markets (wait for confirmed breakout)
- Same conservative position sizing (0.35) to control DD

Why 4h Donchian should beat 4h Supertrend:
- Cleaner breakout signals vs continuous trend following
- Better risk/reward on entry (breakout confirmation)
- Less whipsaw in consolidation periods
"""

import numpy as np
import pandas as pd

name = "donchian_4h_v1"
timeframe = "4h"
leverage = 1.0


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Donchian channel parameters
    period = 20
    
    # Calculate 20-period high and low (rolling max/min)
    # Use proper min_periods to avoid look-ahead
    rolling_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    rolling_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Channel midpoint
    midpoint = (rolling_high + rolling_low) / 2
    
    # Generate signals
    signals = np.zeros(n)
    SIZE = 0.35  # 35% position size - critical for drawdown control
    
    # Track current position state
    # 0 = flat, 1 = long, -1 = short
    position = 0
    
    # First valid index (after period warmup)
    first_valid = period
    
    for i in range(first_valid, n):
        if np.isnan(rolling_high[i]) or np.isnan(rolling_low[i]):
            signals[i] = 0
            continue
        
        prev_close = close[i-1] if i > 0 else close[i]
        curr_close = close[i]
        
        # Long entry: price breaks above 20-period high
        if position <= 0 and curr_close > rolling_high[i-1]:
            position = 1
            signals[i] = SIZE
        
        # Short entry: price breaks below 20-period low
        elif position >= 0 and curr_close < rolling_low[i-1]:
            position = -1
            signals[i] = -SIZE
        
        # Long exit: price crosses below midpoint OR short breakout
        elif position == 1 and curr_close < midpoint[i]:
            position = 0
            signals[i] = 0
        
        # Short exit: price crosses above midpoint OR long breakout
        elif position == -1 and curr_close > midpoint[i]:
            position = 0
            signals[i] = 0
        
        # Hold current position
        else:
            if position == 1:
                signals[i] = SIZE
            elif position == -1:
                signals[i] = -SIZE
            else:
                signals[i] = 0
    
    return signals