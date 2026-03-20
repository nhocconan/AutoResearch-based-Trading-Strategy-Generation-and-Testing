#!/usr/bin/env python3
"""
strategy.py - Volume Momentum V1 with Volatility Regime Filter
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    After v2's success (Sharpe=0.330), testing volume-momentum approach:
    - Volume-weighted price momentum as primary signal
    - Volatility regime filter to avoid choppy markets
    - Multi-period EMA confirmation for trend alignment
    - Volume spike detection for breakout confirmation
    - Conservative position sizing based on ATR
    
    Key differences from v2:
    - Volume-weighted momentum instead of pure price momentum
    - Volatility regime classification (low/med/high)
    - Volume spike confirmation required for entries
    - More conservative signal thresholds
    - Reduced leverage for better risk management

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

name = "volume_momentum_v1"
timeframe = "1h"
leverage = 1.5  # More conservative than v2's 2.0

# EMA periods for trend confirmation
EMA_FAST = 12
EMA_MEDIUM = 26
EMA_SLOW = 50
EMA_TREND = 200

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_THRESHOLD = 1.5  # Volume must be 1.5x average
VOLUME_MA_PERIOD = 20

# Momentum configuration
MOMENTUM_PERIOD = 10
MOMENTUM_THRESHOLD = 0.002  # Minimum momentum to consider

# Volatility regime configuration
ATR_PERIOD = 14
VOLATILITY_LOW_THRESHOLD = 0.005   # Below this = low vol (avoid)
VOLATILITY_HIGH_THRESHOLD = 0.025  # Above this = high vol (reduce size)
VOLATILITY_TARGET = 0.012          # Target volatility for position sizing

# Signal configuration
MIN_SIGNAL = 0.15
MAX_SIGNAL = 0.70
SIGNAL_SMOOTHING = 0.6  # Exponential smoothing factor


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


def calculate_volume_ma(volume: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Calculate volume moving average using only past data.
    
    Args:
        volume: Array of volume values
        period: MA period
    
    Returns:
        Array of volume MA values
    """
    n = len(volume)
    vol_ma = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return vol_ma
    
    volume_series = pd.Series(volume)
    vol_ma_series = volume_series.rolling(window=period, min_periods=period).mean()
    
    vol_ma = np.nan_to_num(vol_ma_series.values, nan=0.0)
    
    return vol_ma


def calculate_momentum(close: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Calculate price momentum (rate of change) using only past data.
    
    Args:
        close: Array of close prices
        period: Momentum period
    
    Returns:
        Array of momentum values (percentage change)
    """
    n = len(close)
    momentum = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return momentum
    
    for i in range(period, n):
        if close[i-period] > 0:
            momentum[i] = (close[i] - close[i-period]) / close[i-period]
    
    return momentum


def calculate_volatility_regime(atr_pct: float) -> str:
    """
    Classify volatility regime based on ATR percentage.
    
    Args:
        atr_pct: ATR as percentage of price
    
    Returns:
        Regime string: "low", "medium", or "high"
    """
    if atr_pct < VOLATILITY_LOW_THRESHOLD:
        return "low"
    elif atr_pct > VOLATILITY_HIGH_THRESHOLD:
        return "high"
    else:
        return "medium"


def calculate_trend_alignment(close: float, ema_fast: float, ema_medium: float, 
                              ema_slow: float, ema_trend: float) -> float:
    """
    Calculate trend alignment score (-1 to 1).
    Positive = bullish alignment, Negative = bearish alignment.
    
    Args:
        close: Current close price
        ema_fast: Fast EMA value
        ema_medium: Medium EMA value
        ema_slow: Slow EMA value
        ema_trend: Long-term trend EMA value
    
    Returns:
        Trend alignment score in range [-1, 1]
    """
    if close <= 0 or ema_trend <= 0:
        return 0.0
    
    # Bullish alignment: close > fast > medium > slow > trend
    bullish_score = 0.0
    if close > ema_fast:
        bullish_score += 0.25
    if ema_fast > ema_medium:
        bullish_score += 0.25
    if ema_medium > ema_slow:
        bullish_score += 0.25
    if ema_slow > ema_trend:
        bullish_score += 0.25
    
    # Bearish alignment: close < fast < medium < slow < trend
    bearish_score = 0.0
    if close < ema_fast:
        bearish_score += 0.25
    if ema_fast < ema_medium:
        bearish_score += 0.25
    if ema_medium < ema_slow:
        bearish_score += 0.25
    if ema_slow < ema_trend:
        bearish_score += 0.25
    
    # Return net alignment score
    return bullish_score - bearish_score


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Volume Momentum V1 Strategy with Volatility Regime Filter.
    
    Signal Logic:
    1. Volume-weighted momentum as primary driver
    2. Trend alignment confirmation from EMA stack
    3. Volume spike confirmation for breakouts
    4. Volatility regime filter for position sizing
    5. Signal smoothing to reduce whipsaws
    
    Entry Conditions:
    - LONG: Positive momentum + bullish trend + volume spike + medium vol
    - SHORT: Negative momentum + bearish trend + volume spike + medium vol
    
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
    volume = np.where(volume <= 0, 1.0, volume)
    
    # Calculate all indicators
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_medium = calculate_ema(close, EMA_MEDIUM)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_trend = calculate_ema(close, EMA_TREND)
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    vol_ma = calculate_volume_ma(volume, VOLUME_MA_PERIOD)
    momentum = calculate_momentum(close, MOMENTUM_PERIOD)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_TREND,
        ATR_PERIOD + 1,
        VOLUME_MA_PERIOD,
        MOMENTUM_PERIOD + 1
    )
    
    # Track previous signal for smoothing
    prev_signal = 0.0
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0 or vol_ma[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate ATR as percentage of price
        atr_pct = atr[i] / close[i]
        
        # Check volatility regime
        vol_regime = calculate_volatility_regime(atr_pct)
        
        # Skip low volatility (choppy market) and reduce size in high volatility
        if vol_regime == "low":
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate volume ratio (current vs average)
        volume_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0.0
        
        # Volume spike confirmation required
        volume_confirmed = volume_ratio >= VOLUME_SPIKE_THRESHOLD
        
        # Calculate trend alignment score
        trend_score = calculate_trend_alignment(
            close[i], ema_fast[i], ema_medium[i], 
            ema_slow[i], ema_trend[i]
        )
        
        # Get momentum value
        mom_value = momentum[i]
        
        # Skip weak momentum
        if abs(mom_value) < MOMENTUM_THRESHOLD:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Determine signal direction
        # Long: positive momentum + bullish trend alignment
        # Short: negative momentum + bearish trend alignment
        if mom_value > 0 and trend_score > 0:
            # LONG signal
            if not volume_confirmed:
                # Reduce signal strength without volume confirmation
                base_signal = 0.4 * trend_score * (mom_value / 0.01)
            else:
                # Full signal with volume confirmation
                base_signal = trend_score * (mom_value / 0.01)
            
            base_signal = min(base_signal, 1.0)  # Cap at 1.0
            
        elif mom_value < 0 and trend_score < 0:
            # SHORT signal
            if not volume_confirmed:
                # Reduce signal strength without volume confirmation
                base_signal = 0.4 * trend_score * (abs(mom_value) / 0.01)
            else:
                # Full signal with volume confirmation
                base_signal = trend_score * (abs(mom_value) / 0.01)
            
            base_signal = max(base_signal, -1.0)  # Cap at -1.0
            
        else:
            # Conflicting signals - stay neutral
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volatility-based position sizing
        if vol_regime == "high":
            vol_factor = 0.5  # Reduce size in high volatility
        else:
            vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
            vol_factor = np.clip(vol_factor, 0.5, 1.5)
        
        raw_signal = base_signal * vol_factor
        
        # Apply exponential smoothing to reduce whipsaws
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        prev_signal = smoothed_signal
        
        # Apply minimum threshold
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals