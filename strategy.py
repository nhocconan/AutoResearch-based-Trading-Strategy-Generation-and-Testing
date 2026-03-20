#!/usr/bin/env python3
"""
strategy.py - Trend Funding Simple V23
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simplified trend-following with funding rate overlay:
    - Primary signal: EMA crossover (12/26) for trend direction
    - Filter: Price above/below 200 EMA for major trend alignment
    - Funding: Extreme funding rates provide contrarian reduction
    - RSI: Basic momentum confirmation (avoid extremes)
    - Minimal filtering to ensure trades actually occur
    
    Why this works:
    - Simpler = less overfitting, more robust
    - Trend following works in crypto but needs position sizing
    - Funding extremes indicate crowded trades to avoid
    - Previous versions had too many filters blocking trades

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

name = "trend_funding_simple_v23"
timeframe = "1h"
leverage = 1.5  # Conservative leverage to reduce drawdown

# EMA configuration for trend detection
EMA_FAST = 12
EMA_SLOW = 26
EMA_MAJOR = 200

# RSI configuration for momentum
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Funding rate configuration
FUNDING_EXTREME = 0.0010  # 0.10% per 8hr
FUNDING_LOOKBACK = 50
FUNDING_IMPACT = 0.30  # How much funding reduces signal

# Signal configuration
MIN_SIGNAL = 0.20  # Minimum signal magnitude to trade
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.30  # EMA smoothing factor for signals


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
    """
    n = len(close)
    if n < period:
        return np.zeros(n, dtype=np.float64)
    
    close_series = pd.Series(close)
    ema_values = close_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    
    return np.nan_to_num(ema_values, nan=0.0)


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using only past data.
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0, dtype=np.float64)
    
    close_series = pd.Series(close)
    delta = close_series.diff()
    
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    avg_gains = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_losses = losses.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gains / avg_losses.replace(0, np.inf)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    
    return np.nan_to_num(rsi_series.values, nan=50.0)


def calculate_funding_zscore(funding_rate: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calculate rolling z-score of funding rate.
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    if n < lookback:
        return np.zeros(n, dtype=np.float64)
    
    funding_series = pd.Series(funding_rate)
    rolling_mean = funding_series.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = funding_series.rolling(window=lookback, min_periods=lookback).std()
    
    zscore = (funding_series - rolling_mean) / rolling_std.replace(0, np.inf)
    
    return np.nan_to_num(zscore.values, nan=0.0)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Funding Simple V23 Strategy.
    
    Signal Logic:
    1. Calculate EMA crossover signal (12/26)
    2. Filter by 200 EMA major trend
    3. Apply RSI momentum confirmation
    4. Reduce signal on extreme funding rates (contrarian)
    5. Smooth signals with EMA
    6. Apply minimum magnitude filter
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, funding_rate, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract price data with error handling
    try:
        close = prices["close"].values.astype(np.float64)
        funding_rate = prices["funding_rate"].values.astype(np.float64)
        funding_rate = np.nan_to_num(funding_rate, nan=0.0)
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Clean data
    close = np.nan_to_num(close, nan=0.0)
    close = np.where(close <= 0, 1.0, close)
    
    # Calculate indicators (all use only past data)
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    rsi = calculate_rsi(close, RSI_PERIOD)
    funding_zscore = calculate_funding_zscore(funding_rate, FUNDING_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(EMA_MAJOR, EMA_SLOW, RSI_PERIOD + 1, FUNDING_LOOKBACK)
    
    # Generate signals
    prev_signal = 0.0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or ema_major[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate EMA crossover signal
        ema_diff = ema_fast[i] - ema_slow[i]
        ema_signal = np.sign(ema_diff) * min(1.0, abs(ema_diff) / close[i] * 100)
        
        # Major trend filter (200 EMA)
        major_trend = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend (reduce signal if conflicting)
        if np.sign(ema_signal) != major_trend and major_trend != 0:
            ema_signal *= 0.5  # Reduce strength on conflict
        
        # RSI momentum confirmation
        rsi_factor = 1.0
        if ema_signal > 0:
            # Long: avoid overbought
            if rsi[i] > RSI_OVERBOUGHT:
                rsi_factor = 0.5
            elif rsi[i] < 30:
                rsi_factor = 0.3  # Weak long in oversold
        elif ema_signal < 0:
            # Short: avoid oversold
            if rsi[i] < RSI_OVERSOLD:
                rsi_factor = 0.5
            elif rsi[i] > 70:
                rsi_factor = 0.3  # Weak short in overbought
        
        # Apply RSI factor
        raw_signal = ema_signal * rsi_factor
        
        # Funding rate contrarian overlay
        # Extreme positive funding (z-score > 2) → reduce long signals
        # Extreme negative funding (z-score < -2) → reduce short signals
        funding_factor = 1.0
        if funding_zscore[i] > 2.0:
            # Extremely positive funding → crowded longs → reduce long bias
            if raw_signal > 0:
                funding_factor = 1.0 - FUNDING_IMPACT
        elif funding_zscore[i] < -2.0:
            # Extremely negative funding → crowded shorts → reduce short bias
            if raw_signal < 0:
                funding_factor = 1.0 - FUNDING_IMPACT
        
        raw_signal *= funding_factor
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals