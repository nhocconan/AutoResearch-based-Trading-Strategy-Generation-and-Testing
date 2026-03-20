#!/usr/bin/env python3
"""
strategy.py - Trend Momentum Simple V21
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simplified trend-following with cleaner signal generation:
    - Primary signal: Fast EMA crossover (12/26) for responsiveness
    - Trend filter: Price above/below 100 EMA for direction bias
    - Funding filter: Only block trades when funding is extreme AGAINST trend
    - Momentum confirmation: RSI in favorable zone (not extreme)
    - Remove volatility normalization that was killing signals
    - Simpler signal combination without over-complication
    
    Why this works:
    - Simpler strategies generalize better across market regimes
    - Funding as filter (not signal) avoids conflicting signals
    - Faster EMA response captures trends earlier
    - Less aggressive filtering ensures actual trades occur
    - Lower leverage reduces drawdown risk

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

name = "trend_momentum_simple_v21"
timeframe = "1h"
leverage = 1.5  # Lower leverage for better risk-adjusted returns

# EMA configuration for trend detection
EMA_FAST = 12
EMA_SLOW = 26
EMA_TREND = 100

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_MIN = 40  # RSI must be above this for longs
RSI_SHORT_MAX = 60  # RSI must be below this for shorts

# Funding rate configuration (FILTER only, not signal)
FUNDING_EXTREME_THRESHOLD = 0.0015  # 0.15% per 8hr = block trades
FUNDING_LOOKBACK = 50  # For calculating recent extremes

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.90  # Maximum signal magnitude
SMOOTHING_FACTOR = 0.30  # EMA smoothing for signals (lower = more responsive)

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.50  # Volume must be at least this % of average


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


def calculate_funding_extremes(funding_rate: np.ndarray, lookback: int = 50) -> tuple:
    """
    Calculate rolling max/min of funding rate.
    Returns: (rolling_max, rolling_min)
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    rolling_max = np.zeros(n, dtype=np.float64)
    rolling_min = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return rolling_max, rolling_min
    
    funding_series = pd.Series(funding_rate)
    rolling_max_series = funding_series.rolling(window=lookback, min_periods=lookback).max()
    rolling_min_series = funding_series.rolling(window=lookback, min_periods=lookback).min()
    
    rolling_max = np.nan_to_num(rolling_max_series.values, nan=0.0)
    rolling_min = np.nan_to_num(rolling_min_series.values, nan=0.0)
    
    return rolling_max, rolling_min


def check_funding_filter(funding_rate: float, 
                         trend_direction: int,
                         funding_max: float,
                         funding_min: float,
                         extreme_threshold: float = 0.0015) -> bool:
    """
    Check if funding rate should block the trade.
    Returns True if trade is ALLOWED, False if blocked.
    
    Logic:
    - If trend is LONG and funding is extremely positive → BLOCK (crowded long)
    - If trend is SHORT and funding is extremely negative → BLOCK (crowded short)
    - Otherwise allow trade
    """
    fr = funding_rate
    
    if trend_direction > 0:  # Long trend
        # Block if funding is extremely positive (crowded longs)
        if fr > extreme_threshold or (funding_max > 0 and fr >= funding_max * 0.85):
            return False
    elif trend_direction < 0:  # Short trend
        # Block if funding is extremely negative (crowded shorts)
        if fr < -extreme_threshold or (funding_min < 0 and fr <= funding_min * 0.85):
            return False
    
    return True  # Trade allowed


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Momentum Simple V21 Strategy.
    
    Signal Logic:
    1. Calculate EMA crossover signal (12/26)
    2. Apply trend filter (price vs 100 EMA)
    3. Check funding filter (block extreme crowded trades)
    4. Apply RSI confirmation
    5. Check volume confirmation
    6. Smooth signals with EMA
    7. Apply minimum magnitude filter
    
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
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Fix invalid prices
    close = np.where(close <= 0, 1.0, close)
    
    # Calculate all indicators (all use only past data)
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_trend = calculate_ema(close, EMA_TREND)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    funding_max, funding_min = calculate_funding_extremes(funding_rate, FUNDING_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_TREND,
        EMA_SLOW,
        RSI_PERIOD + 1,
        VOLUME_LOOKBACK,
        FUNDING_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volume filter (ensure sufficient liquidity)
        if volume_ratio[i] < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate EMA crossover signal
        ema_diff = (ema_fast[i] - ema_slow[i]) / close[i]
        ema_direction = np.sign(ema_diff)
        
        # Skip if EMAs are too close (no clear trend)
        if abs(ema_diff) < 0.0005:  # Less than 0.05% difference
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Trend filter: price vs 100 EMA
        trend_direction = np.sign(close[i] - ema_trend[i])
        
        # Only trade in direction of major trend
        if ema_direction != trend_direction:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Funding filter: block crowded trades
        if not check_funding_filter(funding_rate[i], ema_direction, 
                                     funding_max[i], funding_min[i],
                                     FUNDING_EXTREME_THRESHOLD):
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # RSI confirmation
        rsi_ok = False
        if ema_direction > 0:  # Long
            if rsi[i] >= RSI_LONG_MIN and rsi[i] <= 75:  # Not overbought
                rsi_ok = True
        elif ema_direction < 0:  # Short
            if rsi[i] <= RSI_SHORT_MAX and rsi[i] >= 25:  # Not oversold
                rsi_ok = True
        
        if not rsi_ok:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate raw signal strength based on EMA separation
        signal_strength = min(abs(ema_diff) * 100, 1.0)  # Cap at 1.0
        raw_signal = ema_direction * signal_strength
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals