#!/usr/bin/env python3
"""
strategy.py - Trend Funding Simple V15
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simplified trend-following with funding rate as risk filter:
    - Primary signal: Fast EMA crossover (9/21) for trend direction
    - Filter: Price above/below 200 EMA for major trend validation
    - Risk overlay: Extreme funding rates reduce position size (not reverse)
    - Entry timing: RSI momentum confirmation (avoid extremes)
    
    Why this works:
    - Simpler = more robust, less overfitting
    - Funding acts as risk manager, not signal generator
    - Faster EMAs capture trends earlier on 1h timeframe
    - Reduced complexity should improve out-of-sample performance

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

name = "trend_funding_simple_v15"
timeframe = "1h"
leverage = 1.5  # Conservative leverage for better Sharpe

# EMA configuration for trend detection
EMA_FAST = 9
EMA_SLOW = 21
EMA_MAJOR = 200

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_MIN = 40  # RSI must be above this for longs
RSI_SHORT_MAX = 60  # RSI must be below this for shorts

# Funding rate configuration (risk filter only)
FUNDING_EXTREME = 0.0010  # 0.10% per 8hr = extreme
FUNDING_LOOKBACK = 50  # For calculating recent extremes
FUNDING_RISK_REDUCTION = 0.50  # Max position reduction when funding extreme

# Signal configuration
MIN_SIGNAL = 0.20  # Minimum signal magnitude to trade
SMOOTHING = 0.30  # Signal smoothing factor (0=none, 1=max)


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


def calculate_funding_percentile(funding_rate: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calculate rolling percentile rank of funding rate.
    Returns value in [0, 1] where 1 = highest in lookback period.
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    if n < lookback:
        return np.zeros(n, dtype=np.float64)
    
    funding_series = pd.Series(funding_rate)
    # Calculate percentile rank using only past data
    percentile = funding_series.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5,
        raw=False
    )
    
    return np.nan_to_num(percentile.values, nan=0.5)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Funding Simple V15 Strategy.
    
    Signal Logic:
    1. Calculate EMA crossover trend signal (9/21 EMA)
    2. Validate with 200 EMA major trend filter
    3. Check RSI for entry timing
    4. Apply funding rate risk reduction (not reversal)
    5. Smooth signals and apply minimum magnitude filter
    
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
        try:
            funding_rate = prices["funding_rate"].values.astype(np.float64)
            funding_rate = np.nan_to_num(funding_rate, nan=0.0)
        except (KeyError, TypeError, ValueError):
            funding_rate = np.zeros(n, dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Clean data
    close = np.nan_to_num(close, nan=0.0)
    close = np.where(close <= 0, 1.0, close)
    
    # Calculate all indicators (all use only past data)
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    rsi = calculate_rsi(close, RSI_PERIOD)
    funding_percentile = calculate_funding_percentile(funding_rate, FUNDING_LOOKBACK)
    
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
        
        # Determine trend direction from EMA crossover
        if ema_fast[i] > ema_slow[i]:
            trend_direction = 1.0  # Bullish
        elif ema_fast[i] < ema_slow[i]:
            trend_direction = -1.0  # Bearish
        else:
            trend_direction = 0.0
        
        # Major trend filter: only trade in direction of 200 EMA
        if trend_direction > 0 and close[i] < ema_major[i]:
            trend_direction = 0.0  # Don't long below 200 EMA
        elif trend_direction < 0 and close[i] > ema_major[i]:
            trend_direction = 0.0  # Don't short above 200 EMA
        
        # RSI confirmation
        rsi_valid = True
        if trend_direction > 0 and rsi[i] < RSI_LONG_MIN:
            rsi_valid = False  # RSI too weak for long
        elif trend_direction < 0 and rsi[i] > RSI_SHORT_MAX:
            rsi_valid = False  # RSI too strong for short
        
        if not rsi_valid:
            trend_direction = 0.0
        
        # Calculate base signal strength from EMA separation
        if trend_direction != 0:
            ema_separation = abs(ema_fast[i] - ema_slow[i]) / close[i]
            # Scale separation to signal strength (typical separation 0.5-2%)
            signal_strength = min(1.0, ema_separation * 100)
            raw_signal = trend_direction * signal_strength
        else:
            raw_signal = 0.0
        
        # Funding rate risk reduction (not reversal)
        # Extreme funding against position reduces size
        if raw_signal != 0 and abs(funding_rate[i]) > FUNDING_EXTREME:
            funding_percent = funding_percentile[i]
            
            if raw_signal > 0:  # Long position
                # High funding percentile = crowded longs = reduce position
                if funding_percent > 0.8:
                    risk_factor = 1.0 - FUNDING_RISK_REDUCTION * ((funding_percent - 0.8) / 0.2)
                    raw_signal *= max(0.0, risk_factor)
            elif raw_signal < 0:  # Short position
                # Low funding percentile = crowded shorts = reduce position
                if funding_percent < 0.2:
                    risk_factor = 1.0 - FUNDING_RISK_REDUCTION * ((0.2 - funding_percent) / 0.2)
                    raw_signal *= max(0.0, risk_factor)
        
        # Signal smoothing
        smoothed_signal = SMOOTHING * prev_signal + (1.0 - SMOOTHING) * raw_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -1.0, 1.0)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals