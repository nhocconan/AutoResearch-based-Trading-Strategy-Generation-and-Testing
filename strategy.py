#!/usr/bin/env python3
"""
strategy.py - Trend Momentum V5 Simplified
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Learning from recent failures (#016-#021 all negative Sharpe):
    - Simpler logic outperforms complex multi-factor models
    - Clean trend alignment is more robust than weighted scoring
    - Volatility filtering is critical for risk management
    - Signal smoothing reduces whipsaws effectively
    
    Key changes from v2:
    - Simpler EMA stack alignment (binary + slope confirmation)
    - Cleaner RSI integration (momentum filter, not complex scoring)
    - Better volatility regime detection (ATR percentile)
    - Reduced parameter count to avoid overfitting
    - More conservative signal thresholds

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

name = "trend_momentum_v5_simplified"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for risk management

# EMA periods for trend detection (simplified stack)
EMA_FAST = 12
EMA_MEDIUM = 26
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration (simple momentum filter)
RSI_PERIOD = 14
RSI_LONG_MIN = 45  # Minimum RSI for long entries
RSI_SHORT_MAX = 55  # Maximum RSI for short entries

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_MIN_PERCENTILE = 0.4  # Volume must be above 40th percentile

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_MIN_PCT = 0.003  # Minimum ATR% to trade
VOLATILITY_MAX_PCT = 0.025  # Maximum ATR% to trade
VOLATILITY_TARGET = 0.012  # Target volatility for position sizing

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.15
MAX_SIGNAL_MAGNITUDE = 0.75
SMOOTHING_ALPHA = 0.6  # Exponential smoothing alpha


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
    
    Args:
        close: Array of close prices
        period: EMA period
    
    Returns:
        Array of EMA values
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
    
    Args:
        close: Array of close prices
        period: RSI period
    
    Returns:
        Array of RSI values (0-100)
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
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ATR period
    
    Returns:
        Array of ATR values
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


def calculate_volume_percentile(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume percentile rank using rolling window.
    Only uses past volume data (no look-ahead).
    
    Args:
        volume: Array of volume values
        lookback: Rolling window for percentile calculation
    
    Returns:
        Array of volume percentile ranks (0-1)
    """
    n = len(volume)
    volume_pct = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return volume_pct
    
    volume_series = pd.Series(volume)
    
    for i in range(lookback, n):
        window = volume_series.iloc[i-lookback:i]
        current_vol = volume[i]
        rank = (window < current_vol).sum() / lookback
        volume_pct[i] = rank
    
    return volume_pct


def calculate_atr_percentile(atr_pct: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calculate ATR percentile to detect volatility regime.
    Only uses past ATR data (no look-ahead).
    
    Args:
        atr_pct: Array of ATR as percentage of price
        lookback: Rolling window for percentile calculation
    
    Returns:
        Array of ATR percentile ranks (0-1)
    """
    n = len(atr_pct)
    atr_pct_rank = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return atr_pct_rank
    
    for i in range(lookback, n):
        window = atr_pct[i-lookback:i]
        current = atr_pct[i]
        rank = (window < current).sum() / lookback
        atr_pct_rank[i] = rank
    
    return atr_pct_rank


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Momentum V5 Simplified Strategy.
    
    Signal Logic:
    1. Clean EMA stack alignment for trend direction
    2. EMA slope confirmation (trend strength)
    3. RSI momentum filter (avoid counter-trend entries)
    4. Volume confirmation (liquidity check)
    5. Volatility regime filter (avoid extreme volatility)
    6. Signal smoothing to reduce whipsaws
    
    Entry Conditions:
    - LONG: EMA stack bullish + RSI > 45 + volume confirmed + normal volatility
    - SHORT: EMA stack bearish + RSI < 55 + volume confirmed + normal volatility
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract required columns with safety checks
    try:
        close = prices["close"].values.astype(np.float64)
        high = prices["high"].values.astype(np.float64)
        low = prices["low"].values.astype(np.float64)
        volume = prices["volume"].values.astype(np.float64)
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Handle NaN values
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Ensure valid prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_medium = calculate_ema(close, EMA_MEDIUM)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_pct = calculate_volume_percentile(volume, VOLUME_LOOKBACK)
    
    # Calculate ATR as percentage of price
    atr_pct = np.zeros(n, dtype=np.float64)
    valid_mask = close > 0
    atr_pct[valid_mask] = atr[valid_mask] / close[valid_mask]
    
    # Calculate ATR percentile for volatility regime
    atr_pct_rank = calculate_atr_percentile(atr_pct, 50)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        50  # For ATR percentile
    )
    
    # Track previous signal for smoothing
    prev_signal = 0.0
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volatility regime filter (avoid extreme volatility)
        if atr_pct[i] < VOLATILITY_MIN_PCT or atr_pct[i] > VOLATILITY_MAX_PCT:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volume confirmation
        if volume_pct[i] < VOLUME_MIN_PERCENTILE:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check EMA stack alignment for LONG
        long_alignment = (
            ema_fast[i] > ema_medium[i] and
            ema_medium[i] > ema_slow[i] and
            ema_slow[i] > ema_major[i] and
            close[i] > ema_fast[i]
        )
        
        # Check EMA stack alignment for SHORT
        short_alignment = (
            ema_fast[i] < ema_medium[i] and
            ema_medium[i] < ema_slow[i] and
            ema_slow[i] < ema_major[i] and
            close[i] < ema_fast[i]
        )
        
        # Calculate trend strength (EMA spread normalized by price)
        if long_alignment:
            trend_strength = (ema_fast[i] - ema_major[i]) / close[i]
            trend_direction = 1.0
        elif short_alignment:
            trend_strength = (ema_major[i] - ema_fast[i]) / close[i]
            trend_direction = -1.0
        else:
            # No clear alignment
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Skip weak trends
        if trend_strength < 0.005:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # RSI momentum filter
        if trend_direction > 0:
            # Long: RSI should be above minimum threshold
            if rsi[i] < RSI_LONG_MIN:
                signals[i] = 0.0
                prev_signal = 0.0
                continue
            rsi_factor = min((rsi[i] - RSI_LONG_MIN) / 30.0, 1.0)
        else:
            # Short: RSI should be below maximum threshold
            if rsi[i] > RSI_SHORT_MAX:
                signals[i] = 0.0
                prev_signal = 0.0
                continue
            rsi_factor = min((RSI_SHORT_MAX - rsi[i]) / 30.0, 1.0)
        
        # Volatility-based position sizing (inverse relationship)
        # Reduce position size in higher volatility regimes
        vol_factor = VOLATILITY_TARGET / max(atr_pct[i], 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 1.5)
        
        # Calculate base signal magnitude
        base_signal = trend_direction * trend_strength * 100.0  # Scale to reasonable range
        base_signal = base_signal * rsi_factor * vol_factor
        
        # Apply exponential smoothing to reduce whipsaws
        smoothed_signal = SMOOTHING_ALPHA * prev_signal + (1.0 - SMOOTHING_ALPHA) * base_signal
        prev_signal = smoothed_signal
        
        # Apply minimum threshold
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL_MAGNITUDE, MAX_SIGNAL_MAGNITUDE)
        
        signals[i] = signal
    
    return signals