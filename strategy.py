#!/usr/bin/env python3
"""
EXPERIMENT #004 - MACD Histogram Momentum Strategy (4h)
========================================================
Hypothesis: MACD histogram captures momentum shifts more smoothly than Supertrend's
binary flips. The histogram provides early warning of momentum decay before price
reversal, allowing earlier exits. 4h timeframe filters noise while maintaining
reasonable trade frequency.

Key differences from baseline:
- Momentum-based (MACD histogram) vs trend-following (Supertrend/EMA)
- Gradual momentum decay detection vs hard stop flips
- Same 4h timeframe as best performer (#001)
- Discrete signal levels (0.0, ±0.25, ±0.35) to minimize churning costs
- Histogram threshold filter to avoid whipsaws near zero line
"""

import numpy as np
import pandas as pd

name = "macd_histogram_4h_v1"
timeframe = "4h"
leverage = 1.0


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    n = len(close)
    
    # MACD parameters
    fast_period = 12
    slow_period = 26
    signal_period = 9
    
    # Calculate EMAs using pandas for proper handling
    close_series = pd.Series(close)
    
    # EMA fast and slow
    ema_fast = close_series.ewm(span=fast_period, adjust=False, min_periods=fast_period).mean().values
    ema_slow = close_series.ewm(span=slow_period, adjust=False, min_periods=slow_period).mean().values
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    macd_series = pd.Series(macd_line)
    signal_line = macd_series.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean().values
    
    # Histogram
    histogram = macd_line - signal_line
    
    # Calculate histogram momentum (rate of change of histogram)
    hist_momentum = np.zeros(n)
    hist_momentum[1:] = np.diff(histogram)
    
    # Generate signals with discrete position sizing
    signals = np.zeros(n)
    
    # Position size levels - discrete to minimize churning
    SIZE_MEDIUM = 0.25  # 25% position
    SIZE_LARGE = 0.35   # 35% position
    
    # Threshold to avoid whipsaws near zero
    HIST_THRESHOLD = 0.0
    
    # Track previous signal to avoid unnecessary flips
    prev_signal = 0.0
    
    # Find first valid index (after all warmup periods)
    first_valid = max(fast_period, slow_period, signal_period) + 1
    
    for i in range(first_valid, n):
        if np.isnan(histogram[i]) or np.isnan(hist_momentum[i]):
            signals[i] = 0.0
            continue
        
        # Determine signal based on histogram position and momentum
        if histogram[i] > HIST_THRESHOLD:
            # Bullish territory
            if hist_momentum[i] >= 0:
                # Momentum increasing or stable - full position
                signals[i] = SIZE_LARGE
            else:
                # Momentum decreasing but still positive - reduce position
                signals[i] = SIZE_MEDIUM
        elif histogram[i] < -HIST_THRESHOLD:
            # Bearish territory
            if hist_momentum[i] <= 0:
                # Momentum decreasing or stable - full short
                signals[i] = -SIZE_LARGE
            else:
                # Momentum increasing but still negative - reduce short
                signals[i] = -SIZE_MEDIUM
        else:
            # Near zero - flat
            signals[i] = 0.0
        
        # Apply hysteresis to reduce churning
        # Only flip if signal magnitude changes significantly
        if abs(prev_signal) > 0 and abs(signals[i]) > 0:
            # Same direction - keep
            if np.sign(prev_signal) == np.sign(signals[i]):
                signals[i] = prev_signal  # Maintain previous to avoid churn
        elif abs(signals[i]) < SIZE_MEDIUM and abs(prev_signal) >= SIZE_MEDIUM:
            # Reducing position - allow it (momentum decay)
            pass
        elif abs(signals[i]) == 0 and abs(prev_signal) > 0:
            # Going flat - allow it (exit signal)
            pass
        
        prev_signal = signals[i]
    
    return signals