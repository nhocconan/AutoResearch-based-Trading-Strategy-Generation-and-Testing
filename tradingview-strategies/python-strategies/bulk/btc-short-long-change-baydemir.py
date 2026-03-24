#!/usr/bin/env python3
import numpy as np
import pandas as pd

name = "BTC Short/Long Change Strategy [Baydemir] [Adapted]"
timeframe = "1d"
leverage = 1

def calculate_wma(data, length):
    """Calculate Weighted Moving Average using numpy."""
    if len(data) < length:
        return np.full_like(data, np.nan, dtype=np.float64)
    
    weights = np.arange(1, length + 1, dtype=np.float64)
    # Convolve returns length N + M - 1. Mode valid returns N - M + 1.
    raw_wma = np.convolve(data, weights, mode='valid') / np.sum(weights)
    
    # Pad the beginning with NaNs to match original length
    padding = np.full(length - 1, np.nan, dtype=np.float64)
    return np.concatenate([padding, raw_wma])

def generate_signals(prices):
    """
    Generate trading signals based on WMA crossover logic.
    
    Note: Original strategy uses external tickers 'btcusdshorts' and 'btcusdlongs'.
    These are unavailable in standard OHLCV data. This implementation substitutes
    'volume' as a proxy for net long/short flow to maintain algorithmic structure.
    
    Args:
        prices (pd.DataFrame): DataFrame with columns ['open_time', 'open', 'high', 'low', 'close', 'volume'].
        
    Returns:
        np.ndarray: Array of signals (1.0 for Long, -1.0 for Short, 0.0 for Neutral).
    """
    if prices.empty:
        return np.array([], dtype=np.float64)
    
    # Parameters
    len2 = 5  # SLOW PERIOD from Pine Script input
    
    # Extract data
    # Proxy for 'hist = -line1 + line2' (Net Volume) using available 'volume' column
    hist = prices['volume'].values.astype(np.float64)
    
    # Calculate WMA of hist
    mahist = calculate_wma(hist, len2)
    
    # Initialize signals array
    signals = np.zeros(len(prices), dtype=np.float64)
    
    # Calculate Crossover (Long) and Crossunder (Short) conditions
    # Crossover: hist crosses above mahist
    # Current: hist > mahist
    # Previous: hist <= mahist
    
    curr_above = hist > mahist
    curr_below = hist < mahist
    
    # Shift arrays to get previous values
    prev_hist = np.roll(hist, 1)
    prev_mahist = np.roll(mahist, 1)
    
    # Handle first bar (index 0) where shift brings last element
    # Set first element comparisons to False to avoid lookahead/wraparound
    prev_hist[0] = np.nan
    prev_mahist[0] = np.nan
    
    prev_below_or_equal = prev_hist <= prev_mahist
    prev_above_or_equal = prev_hist >= prev_mahist
    
    # Conditions
    long_condition = curr_above & prev_below_or_equal
    short_condition = curr_below & prev_above_or_equal
    
    # Handle NaNs in conditions (NaN comparisons result in False, which is safe)
    # Assign signals
    signals[long_condition] = 1.0
    signals[short_condition] = -1.0
    
    return signals
