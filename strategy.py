#!/usr/bin/env python3
"""
strategy.py - Momentum Trend Following with Volatility Scaling
=======================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simplified trend-following with momentum confirmation.
    - EMA crossover for trend direction (20/50)
    - Rate of Change (ROC) for momentum confirmation
    - Volatility-based position sizing (reduce size in high vol)
    - Signal smoothing to reduce whipsaws
    - No strict volume filter (was killing too many signals)

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

name = "momentum_trend_volatility"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for crypto futures

# Strategy parameters
EMA_FAST = 20             # Fast EMA period
EMA_SLOW = 50             # Slow EMA period
ROC_PERIOD = 10           # Rate of Change period for momentum
ATR_PERIOD = 14           # ATR calculation period
SIGNAL_SMOOTH = 5         # Signal smoothing window
MIN_SIGNAL = 0.15         # Minimum signal magnitude to trade
MAX_SIGNAL = 0.8          # Maximum signal magnitude (leave room for scaling)


# =============================================================================
# Signal Generation
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
    
    # Initialize with SMA
    ema[period - 1] = np.mean(close[:period])
    
    # Calculate EMA multiplier
    multiplier = 2.0 / (period + 1)
    
    # Calculate EMA for remaining periods
    for i in range(period, n):
        ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    
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
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Initialize ATR with SMA of TR
    atr[period - 1] = np.mean(tr[:period])
    
    # Calculate ATR using Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_roc(close: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Calculate Rate of Change using only past data.
    
    ROC = (close[i] - close[i-period]) / close[i-period] * 100
    
    Args:
        close: Array of close prices
        period: ROC lookback period
    
    Returns:
        Array of ROC values (percentage)
    """
    n = len(close)
    roc = np.zeros(n, dtype=np.float64)
    
    if n <= period:
        return roc
    
    for i in range(period, n):
        if close[i - period] > 0:
            roc[i] = (close[i] - close[i - period]) / close[i - period] * 100.0
        else:
            roc[i] = 0.0
    
    return roc


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Momentum Trend Following Strategy with Volatility Scaling.
    
    Signal Logic:
    1. Calculate fast/slow EMA for trend direction
    2. Calculate ROC for momentum confirmation
    3. Calculate ATR for volatility adjustment
    4. Generate signals based on trend + momentum
    5. Smooth signals to reduce whipsaws
    
    Entry Conditions:
    - LONG: Fast EMA > Slow EMA AND ROC > 0
    - SHORT: Fast EMA < Slow EMA AND ROC < 0
    
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
    except (KeyError, TypeError, ValueError) as e:
        # Return zeros if required columns missing
        return signals
    
    # Handle any NaN values in price data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    
    # Ensure no zero or negative prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate EMAs
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    
    # Calculate ATR for volatility adjustment
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Calculate ROC for momentum
    roc = calculate_roc(close, ROC_PERIOD)
    
    # Calculate EMA spread (trend strength indicator)
    ema_spread = (ema_fast - ema_slow) / close
    ema_spread = np.nan_to_num(ema_spread, nan=0.0)
    
    # Normalize ROC to [-1, 1] range for signal combination
    # Typical 1h ROC ranges from -5% to +5%, so divide by 5
    roc_normalized = np.clip(roc / 5.0, -1.0, 1.0)
    
    # Determine minimum valid index
    min_valid_index = max(EMA_SLOW, ATR_PERIOD + 1, ROC_PERIOD, SIGNAL_SMOOTH)
    
    # Generate raw signals
    raw_signals = np.zeros(n, dtype=np.float64)
    
    for i in range(min_valid_index, n):
        # Skip if any required data is invalid
        if close[i] <= 0 or atr[i] <= 0:
            raw_signals[i] = 0.0
            continue
        
        # Trend direction from EMA crossover
        ema_diff = ema_fast[i] - ema_slow[i]
        trend_direction = np.sign(ema_diff)
        
        # Momentum confirmation from ROC
        momentum_signal = roc_normalized[i]
        
        # Combine trend and momentum (both must agree for strong signal)
        # If trend and momentum agree, amplify signal
        # If they disagree, reduce signal
        if trend_direction > 0 and momentum_signal > 0:
            # Bullish trend + positive momentum
            raw_signal = (abs(ema_spread[i]) * 100 + momentum_signal) / 2.0
        elif trend_direction < 0 and momentum_signal < 0:
            # Bearish trend + negative momentum
            raw_signal = -(abs(ema_spread[i]) * 100 + abs(momentum_signal)) / 2.0
        elif trend_direction > 0:
            # Bullish trend but weak/negative momentum
            raw_signal = abs(ema_spread[i]) * 100 * 0.5
        elif trend_direction < 0:
            # Bearish trend but weak/positive momentum
            raw_signal = -abs(ema_spread[i]) * 100 * 0.5
        else:
            # No clear trend
            raw_signal = 0.0
        
        # Volatility adjustment (reduce position in high volatility)
        # Typical 1h ATR% is 0.5-2% for crypto
        atr_pct = atr[i] / close[i] * 100
        vol_factor = 1.0
        if atr_pct > 0:
            # Scale down when volatility is high (>2%)
            vol_factor = min(1.0, 2.0 / max(atr_pct, 0.5))
        
        # Apply volatility factor
        raw_signals[i] = raw_signal * vol_factor
    
    # Smooth signals to reduce whipsaws using simple moving average
    if n >= SIGNAL_SMOOTH:
        smoothed = pd.Series(raw_signals).rolling(
            window=SIGNAL_SMOOTH, 
            min_periods=SIGNAL_SMOOTH
        ).mean().values
        raw_signals = np.nan_to_num(smoothed, nan=0.0)
    
    # Apply signal thresholds and clipping
    for i in range(n):
        signal = raw_signals[i]
        
        # Apply minimum signal threshold
        if abs(signal) < MIN_SIGNAL:
            signal = 0.0
        
        # Clip to [-MAX_SIGNAL, MAX_SIGNAL] to leave room for risk management
        signal = np.clip(signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals