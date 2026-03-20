#!/usr/bin/env python3
"""
strategy.py - Trend Funding Hybrid V17
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simplified trend-following with funding rate contrarian overlay:
    - Primary signal: EMA crossover (12/26) for trend direction
    - Filter: Price vs 100 EMA for major trend validation
    - RSI filter: Avoid entering at extreme overbought/oversold levels
    - Funding overlay: Extreme funding rates provide contrarian signal
    - Signal smoothing: EMA on signals to reduce whipsaws
    
    Why this works:
    - Simpler than v16, fewer parameters to overfit
    - Funding only acts on true extremes (contrarian)
    - Removed volume/volatility filters that were killing trades
    - Better signal combination logic

Look-Ahead Safety:
    - All rolling calculations use only past data (min_periods respected)
    - No .shift(-n) or future index access
    - Signal at bar t uses only prices.iloc[:t+1]
"""

import numpy as np
import pandas as pd

# =============================================================================
# Strategy Configuration
# =============================================================================

name = "trend_funding_hybrid_v17"
timeframe = "1h"
leverage = 2.0  # Conservative leverage

# EMA configuration for trend detection
EMA_FAST = 12
EMA_SLOW = 26
EMA_MAJOR = 100

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Funding rate configuration
FUNDING_EXTREME = 0.0008  # 0.08% per 8hr = extreme
FUNDING_WEIGHT = 0.30  # How much funding affects signal
FUNDING_LOOKBACK = 100  # For calculating extremes

# Signal configuration
SMOOTHING = 0.60  # EMA smoothing for signals (0=none, 1=max)
MIN_SIGNAL = 0.20  # Minimum signal magnitude to trade
MAX_SIGNAL = 0.80  # Maximum signal magnitude


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Funding Hybrid V17 Strategy.
    
    Signal Logic:
    1. Calculate EMA crossover for trend direction
    2. Validate with major EMA filter
    3. Apply RSI filter to avoid extremes
    4. Add funding rate contrarian overlay on extremes
    5. Smooth signals to reduce whipsaws
    6. Apply minimum magnitude filter
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, funding_rate, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Need enough data for all indicators
    min_bars = max(EMA_MAJOR, EMA_SLOW, RSI_PERIOD + 1, FUNDING_LOOKBACK) + 10
    if n < min_bars:
        return signals
    
    # Extract price data with error handling
    try:
        close = prices["close"].values.astype(np.float64)
        high = prices["high"].values.astype(np.float64)
        low = prices["low"].values.astype(np.float64)
        
        try:
            funding_rate = prices["funding_rate"].values.astype(np.float64)
            funding_rate = np.nan_to_num(funding_rate, nan=0.0)
        except (KeyError, TypeError, ValueError):
            funding_rate = np.zeros(n, dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Clean data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    
    # Fix invalid prices (only using current/past data)
    for i in range(n):
        if close[i] <= 0:
            close[i] = 1.0
        if high[i] <= 0 or high[i] < close[i]:
            high[i] = close[i] * 1.01
        if low[i] <= 0 or low[i] > close[i]:
            low[i] = close[i] * 0.99
    
    # Calculate indicators using pandas (respects min_periods, no look-ahead)
    close_series = pd.Series(close)
    
    # EMAs - only uses past data
    ema_fast = close_series.ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    ema_slow = close_series.ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    ema_major = close_series.ewm(span=EMA_MAJOR, adjust=False, min_periods=EMA_MAJOR).mean().values
    
    # RSI - only uses past data
    delta = close_series.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    avg_gains = gains.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    avg_losses = losses.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    rs = avg_gains / avg_losses.replace(0, np.inf)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = np.nan_to_num(rsi.values, nan=50.0)
    
    # Funding rate extremes - rolling percentile only uses past data
    funding_series = pd.Series(funding_rate)
    funding_90 = funding_series.rolling(window=FUNDING_LOOKBACK, min_periods=FUNDING_LOOKBACK).quantile(0.90).values
    funding_10 = funding_series.rolling(window=FUNDING_LOOKBACK, min_periods=FUNDING_LOOKBACK).quantile(0.10).values
    funding_90 = np.nan_to_num(funding_90, nan=0.0)
    funding_10 = np.nan_to_num(funding_10, nan=0.0)
    
    # Generate signals
    prev_signal = 0.0
    
    for i in range(min_bars, n):
        # Skip invalid bars
        if close[i] <= 0 or ema_major[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Trend direction from EMA crossover
        ema_diff = ema_fast[i] - ema_slow[i]
        trend_dir = np.sign(ema_diff)
        
        # Major trend filter (price vs 100 EMA)
        major_dir = np.sign(close[i] - ema_major[i])
        
        # Only trade with major trend direction
        if trend_dir != major_dir and trend_dir != 0:
            # Conflicting signals → reduce strength
            trend_strength = 0.5
        else:
            # Aligned signals → full strength
            trend_strength = 1.0
        
        # RSI filter to avoid extreme entries
        rsi_factor = 1.0
        if trend_dir > 0 and rsi[i] > RSI_OVERBOUGHT:
            rsi_factor = 0.5  # Reduce long strength if overbought
        elif trend_dir < 0 and rsi[i] < RSI_OVERSOLD:
            rsi_factor = 0.5  # Reduce short strength if oversold
        
        # Base trend signal
        trend_signal = trend_dir * trend_strength * rsi_factor
        
        # Funding rate contrarian overlay
        funding_signal = 0.0
        fr = funding_rate[i]
        
        # Extreme positive funding → short bias (contrarian)
        if fr > FUNDING_EXTREME or (funding_90[i] > 0 and fr >= funding_90[i] * 0.9):
            funding_signal = -FUNDING_WEIGHT
        
        # Extreme negative funding → long bias (contrarian)
        elif fr < -FUNDING_EXTREME or (funding_10[i] < 0 and fr <= funding_10[i] * 0.9):
            funding_signal = FUNDING_WEIGHT
        
        # Combine signals
        if abs(trend_signal) > 0.3 and abs(funding_signal) > 0.1:
            if np.sign(trend_signal) != np.sign(funding_signal):
                # Conflict: funding contradicts trend → reduce strength
                raw_signal = trend_signal * 0.7 + funding_signal
            else:
                # Aligned: both agree → reinforce slightly
                raw_signal = trend_signal * 0.75 + funding_signal * 0.25
        else:
            # Weak signals or no funding extreme
            raw_signal = trend_signal * 0.8 + funding_signal * 0.2
        
        # Signal smoothing (EMA on signals to reduce whipsaws)
        smoothed_signal = SMOOTHING * prev_signal + (1.0 - SMOOTHING) * raw_signal
        
        # Minimum magnitude filter (no position if signal too weak)
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to max signal magnitude
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals