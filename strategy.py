#!/usr/bin/env python3
"""
EXPERIMENT #009 - Regime-Adaptive Supertrend (4h)
=================================================
Hypothesis: Adapting Supertrend multiplier based on volatility regime will reduce
whipsaws during choppy periods while maintaining trend exposure during volatile trends.

Key improvements over supertrend_4h_v1:
- Bollinger Band Width percentile to detect volatility regime
- High vol regime (BW > 60th pct): tighter Supertrend (mult=2.5) for faster entries
- Low vol regime (BW < 40th pct): wider Supertrend (mult=3.5) or flat to avoid chop
- Medium vol: standard Supertrend (mult=3.0)
- Same conservative position sizing (0.35) to control DD
- Discrete signal levels to minimize churning costs

This builds on the only winning strategy (Supertrend 4h) but adds regime filtering
to improve Sharpe ratio by avoiding bad trades in choppy markets.
"""

import numpy as np
import pandas as pd

name = "regime_supertrend_4h_v1"
timeframe = "4h"
leverage = 1.0


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    if n < 100:
        return np.zeros(n)
    
    # Calculate True Range
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], 
                    abs(high[i] - close[i-1]), 
                    abs(low[i] - close[i-1]))
    
    # ATR(10)
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Bollinger Bands for regime detection (20 period, 2 std)
    bb_window = 20
    bb_sma = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).mean().values
    bb_std = pd.Series(close).rolling(window=bb_window, min_periods=bb_window).std().values
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std
    
    # Bollinger Band Width (normalized)
    bb_width = (bb_upper - bb_lower) / bb_sma
    
    # Calculate rolling percentile of BB Width (60-period lookback)
    regime_window = 60
    bb_width_percentile = np.zeros(n)
    for i in range(regime_window, n):
        if np.isnan(bb_width[i]):
            bb_width_percentile[i] = np.nan
            continue
        lookback = bb_width[i-regime_window:i+1]
        lookback = lookback[~np.isnan(lookback)]
        if len(lookback) > 0:
            bb_width_percentile[i] = np.sum(lookback <= bb_width[i]) / len(lookback)
        else:
            bb_width_percentile[i] = 0.5
    
    # Supertrend calculation with regime-adaptive multiplier
    supertrend = np.zeros(n)
    trend_direction = np.zeros(n)  # 1 = long, -1 = short, 0 = flat
    
    # Initialize
    first_valid = max(10, bb_window, regime_window)
    
    for i in range(first_valid, n):
        if np.isnan(atr[i]) or np.isnan(bb_width_percentile[i]):
            if i > 0:
                supertrend[i] = supertrend[i-1]
                trend_direction[i] = trend_direction[i-1]
            continue
        
        # Regime-adaptive multiplier
        percentile = bb_width_percentile[i]
        if percentile > 0.60:
            # High volatility regime - tighter stop for faster trend capture
            multiplier = 2.5
            base_size = 0.35
        elif percentile < 0.40:
            # Low volatility regime - wider stop or flat to avoid chop
            multiplier = 3.5
            base_size = 0.25  # Reduce size in chop
        else:
            # Medium volatility - standard
            multiplier = 3.0
            base_size = 0.35
        
        # Calculate bands
        hl2 = (high[i] + low[i]) / 2
        upper_band = hl2 + multiplier * atr[i]
        lower_band = hl2 - multiplier * atr[i]
        
        if i == first_valid:
            supertrend[i] = upper_band
            trend_direction[i] = -1
        else:
            # If previous trend was long
            if trend_direction[i-1] == 1:
                if close[i] > supertrend[i-1]:
                    # Stay long
                    supertrend[i] = max(lower_band, supertrend[i-1])
                    trend_direction[i] = 1
                else:
                    # Flip to short
                    supertrend[i] = upper_band
                    trend_direction[i] = -1
            elif trend_direction[i-1] == -1:
                # Previous trend was short
                if close[i] < supertrend[i-1]:
                    # Stay short
                    supertrend[i] = min(upper_band, supertrend[i-1])
                    trend_direction[i] = -1
                else:
                    # Flip to long
                    supertrend[i] = lower_band
                    trend_direction[i] = 1
            else:
                # Previous was flat - initialize based on price position
                if close[i] > hl2:
                    supertrend[i] = lower_band
                    trend_direction[i] = 1
                else:
                    supertrend[i] = upper_band
                    trend_direction[i] = -1
    
    # Generate signals with discrete position sizing
    signals = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(trend_direction[i]):
            signals[i] = 0.0
        elif trend_direction[i] == 1:
            # Long signal - use size based on regime
            percentile = bb_width_percentile[i]
            if percentile > 0.60:
                signals[i] = 0.35
            elif percentile < 0.40:
                signals[i] = 0.25
            else:
                signals[i] = 0.35
        elif trend_direction[i] == -1:
            # Short signal
            percentile = bb_width_percentile[i]
            if percentile > 0.60:
                signals[i] = -0.35
            elif percentile < 0.40:
                signals[i] = -0.25
            else:
                signals[i] = -0.35
    
    return signals