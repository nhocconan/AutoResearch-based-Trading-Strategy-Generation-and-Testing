#!/usr/bin/env python3
import pandas as pd
import numpy as np

name = "BTC 3X PAIR/HULL MA (Adapted)"
timeframe = "1h"
leverage = 3

def wma(series, window):
    """Calculate Weighted Moving Average using numpy convolution."""
    weights = np.arange(1, window + 1)
    conv = np.convolve(series, weights, 'valid')
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, conv / weights.sum()])

def calculate_hull(series, length):
    """Calculate Hull Moving Average manually."""
    half = int(round(length / 2))
    sqrt_len = int(round(np.sqrt(length)))
    wma_half = wma(series, half)
    wma_full = wma(series, length)
    raw_hull = 2 * wma_half - wma_full
    hull = wma(raw_hull, sqrt_len)
    return hull

def generate_signals(prices):
    """
    Generate trading signals based on Hull Moving Average logic.
    Adapts multi-symbol Pine Script to single-symbol Python logic.
    """
    if prices is None or len(prices) == 0:
        return np.array([])

    # Ensure DataFrame
    df = prices.copy() if isinstance(prices, pd.DataFrame) else pd.DataFrame(prices)
    
    # Required columns
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate OHLC4 source shifted by 1 bar (matches Pine: src = ohlc4[1])
    ohlc4 = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
    src = ohlc4.shift(1)
    
    # Hull MA Parameters (from Pine: keh=7)
    keh = 7
    
    # Calculate Hull MA on shifted source
    # Note: wma returns numpy array, align index
    src_values = src.values
    hull_values = calculate_hull(src_values, keh)
    hull = pd.Series(hull_values, index=df.index)
    
    # Calculate Momentum (Proxy for Daily Confidence)
    # Pine: confidence = (daily_src - daily_src[1]) / daily_src[1]
    # Adaptation: 24-period return on 1h data
    momentum = df['close'].pct_change(24)
    
    # Thresholds
    dt = 0.001
    
    # Initialize signals array (0: Neutral, 1: Long, -1: Short)
    signals = np.zeros(len(df))
    
    # Calculate Hull Slope
    hull_shift = hull.shift(1)
    hull_up = hull > hull_shift
    hull_down = hull < hull_shift
    
    # Entry/Exit Conditions (Adapted from Pine)
    # Original Long: e1<e2 and n1>n2 and s1>s2 and confidence>dt
    # Adapted Long: Hull Up and Momentum > dt
    long_condition = hull_up.values & (momentum.values > dt)
    
    # Original Short: e1>e2 and n1<n2 and s1<s2 and confidence<dt
    # Adapted Short: Hull Down and Momentum < -dt
    short_condition = hull_down.values & (momentum.values < -dt)
    
    # Assign signals
    # Priority: Long > Short > Neutral
    signals[long_condition] = 1
    signals[short_condition] = -1
    
    # Handle NaNs from rolling windows
    signals[np.isnan(hull_values)] = 0
    
    return signals.astype(int)