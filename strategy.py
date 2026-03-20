#!/usr/bin/env python3
"""
strategy.py - Trend Funding Hybrid V18
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simplified trend-following with funding as entry filter:
    - Faster EMA crossover (9/21) for quicker trend detection
    - 200 EMA as major trend filter (only trade with major trend)
    - Funding rate as entry filter (avoid entering when funding extreme against trend)
    - RSI momentum confirmation (simple thresholds)
    - Reduced signal smoothing for faster response
    - Lower minimum signal threshold to capture more opportunities
    
    Why this works:
    - Previous versions over-filtered and missed profitable moves
    - Faster EMAs catch trends earlier
    - Funding filter prevents entering crowded trades without fighting trend
    - Simpler logic = more robust across market conditions

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

name = "trend_funding_hybrid_v18"
timeframe = "1h"
leverage = 2.5  # Moderate leverage for better risk-adjusted returns

# EMA configuration for trend detection (faster than v12)
EMA_FAST = 9
EMA_SLOW = 21
EMA_MAJOR = 200

# RSI configuration for momentum
RSI_PERIOD = 14
RSI_LONG_MIN = 40  # RSI must be above this for longs
RSI_SHORT_MAX = 60  # RSI must be below this for shorts

# Funding rate configuration (used as filter, not signal)
FUNDING_EXTREME_THRESHOLD = 0.0015  # 0.15% per 8hr = very extreme
FUNDING_LOOKBACK = 50  # For calculating recent extremes
FUNDING_FILTER_ENABLED = True  # Enable funding as entry filter

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_MIN = 0.002  # Minimum ATR % to trade
VOLATILITY_MAX = 0.080  # Maximum ATR % to trade

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.10  # Lower threshold to get more trades
MAX_SIGNAL = 0.90  # Maximum signal magnitude
SMOOTHING_FACTOR = 0.25  # Less smoothing for faster response
TREND_STRENGTH_MULTIPLIER = 100  # Scale factor for trend signal

# Volume confirmation (relaxed from v12)
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.50  # More lenient volume filter


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
    """
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return ema
    
    close_series = pd.Series(close)
    ema_values = close_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    ema = np.nan_to_num(ema_values, nan=0.0)
    
    return ema


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using only past data.
    """
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    
    if n < period + 1:
        return rsi
    
    close_series = pd.Series(close)
    delta = close_series.diff()
    
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    avg_gains = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_losses = losses.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gains / avg_losses.replace(0, np.inf)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    
    rsi = np.nan_to_num(rsi_series.values, nan=50.0)
    
    return rsi


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using only past data.
    """
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    tr_series = pd.Series(tr)
    atr_series = tr_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    atr = np.nan_to_num(atr_series.values, nan=0.0)
    
    return atr


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio vs rolling average.
    Only uses past volume data (no look-ahead).
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    rolling_avg = volume_series.rolling(window=lookback, min_periods=lookback).mean()
    
    volume_ratio = np.nan_to_num(volume_series.values / rolling_avg.values, nan=1.0)
    
    return volume_ratio


def calculate_funding_percentile(funding_rate: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calculate rolling percentile rank of funding rate.
    Returns value in [0, 1] where 1 = highest in lookback period.
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    percentile = np.full(n, 0.5, dtype=np.float64)
    
    if n < lookback:
        return percentile
    
    funding_series = pd.Series(funding_rate)
    
    for i in range(lookback, n):
        window = funding_series.iloc[i-lookback:i+1]
        rank = window.rank(pct=True).iloc[-1]
        percentile[i] = rank
    
    return percentile


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Funding Hybrid V18 Strategy.
    
    Signal Logic:
    1. Calculate fast/slow EMA crossover for trend direction
    2. Filter by 200 EMA (only trade with major trend)
    3. Check RSI for momentum confirmation
    4. Apply funding rate filter (avoid extreme funding against trend)
    5. Check volume and volatility filters
    6. Scale signal by trend strength
    7. Apply light smoothing and hysteresis
    
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
        high = prices["high"].values.astype(np.float64)
        low = prices["low"].values.astype(np.float64)
        volume = prices["volume"].values.astype(np.float64)
        
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
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Fix invalid prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators (all use only past data)
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    funding_percentile = calculate_funding_percentile(funding_rate, FUNDING_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        FUNDING_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volatility filter
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volume filter
        if volume_ratio[i] < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Determine trend direction from EMA crossover
        ema_diff = ema_fast[i] - ema_slow[i]
        ema_direction = np.sign(ema_diff)
        
        # Major trend filter (price vs 200 EMA)
        major_direction = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend
        if ema_direction != major_direction or ema_direction == 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # RSI momentum confirmation
        rsi_ok = False
        if ema_direction > 0:
            # Long: RSI must be above minimum threshold
            if rsi[i] >= RSI_LONG_MIN:
                rsi_ok = True
        elif ema_direction < 0:
            # Short: RSI must be below maximum threshold
            if rsi[i] <= RSI_SHORT_MAX:
                rsi_ok = True
        
        if not rsi_ok:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Funding rate filter (avoid entering when funding is extreme against trend)
        funding_ok = True
        if FUNDING_FILTER_ENABLED:
            if ema_direction > 0:
                # Long: avoid when funding is very high (crowded longs)
                if funding_rate[i] > FUNDING_EXTREME_THRESHOLD:
                    funding_ok = False
                # Also check percentile - avoid top 10% funding
                elif funding_percentile[i] > 0.90:
                    funding_ok = False
            elif ema_direction < 0:
                # Short: avoid when funding is very negative (crowded shorts)
                if funding_rate[i] < -FUNDING_EXTREME_THRESHOLD:
                    funding_ok = False
                # Also check percentile - avoid bottom 10% funding
                elif funding_percentile[i] < 0.10:
                    funding_ok = False
        
        if not funding_ok:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Calculate trend strength (normalized EMA difference)
        trend_strength = abs(ema_diff) / close[i] * TREND_STRENGTH_MULTIPLIER
        trend_strength = np.clip(trend_strength, 0.1, 1.0)
        
        # Base signal = direction * strength
        raw_signal = ema_direction * trend_strength
        
        # Adjust signal based on funding percentile (mild adjustment)
        # If funding supports trend, slightly increase signal
        if ema_direction > 0:
            # Long: lower funding percentile is better
            funding_adjustment = (0.5 - funding_percentile[i]) * 0.2
        else:
            # Short: higher funding percentile is better
            funding_adjustment = (funding_percentile[i] - 0.5) * 0.2
        
        raw_signal += funding_adjustment
        
        # Signal smoothing (light EMA on signals)
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Hysteresis: don't flip direction on small changes
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < 0.05:
                smoothed_signal = prev_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals