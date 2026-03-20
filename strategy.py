#!/usr/bin/env python3
"""
strategy.py - Trend Pullback with RSI Mean Reversion
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Trend-following with mean-reversion entries performs better than pure
    trend-following or pure mean-reversion. The key insight:
    
    - Identify the primary trend using long-term EMA (100-period)
    - Enter on pullbacks within the trend using RSI extremes
    - LONG: Price above EMA100 + RSI dips to 35-45 (buy the dip)
    - SHORT: Price below EMA100 + RSI rallies to 55-65 (sell the rally)
    
    Key improvement over #002:
    - Simpler logic with fewer conflicting filters
    - RSI used for entry timing (mean-reversion within trend) not momentum
    - Removed volume filter (often noisy on crypto)
    - Removed BB filter (redundant with RSI for mean-reversion)
    - More aggressive signal magnitudes when conditions align

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

name = "trend_pullback_rsi"
timeframe = "1h"
leverage = 2.5  # Moderate leverage, confident in trend filter

# Strategy parameters - simplified from #002
EMA_TREND = 100                   # Long-term EMA for trend direction
EMA_ENTRY = 21                    # Shorter EMA for entry timing
RSI_PERIOD = 14                   # RSI calculation period
RSI_LONG_ENTRY = 40               # RSI level for long entries (buy dip)
RSI_LONG_RANGE = 10               # Acceptable range around entry (35-45)
RSI_SHORT_ENTRY = 60              # RSI level for short entries (sell rally)
RSI_SHORT_RANGE = 10              # Acceptable range around entry (55-65)
ATR_PERIOD = 14                   # ATR calculation period
VOLATILITY_TARGET = 0.015         # Target hourly volatility for position sizing
MIN_SIGNAL = 0.25                 # Minimum signal magnitude to trade
MAX_SIGNAL = 0.85                 # Maximum signal magnitude
TREND_THRESHOLD = 0.002           # Minimum price/EMA separation for valid trend
RSI_STRENGTH_MIN = 0.6            # Minimum RSI quality score


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


def calculate_rsi_quality(rsi: np.ndarray, target: float, tolerance: float) -> np.ndarray:
    """
    Calculate RSI quality score based on proximity to target level.
    Higher score = better entry opportunity.
    
    Args:
        rsi: Array of RSI values
        target: Target RSI level for entry
        tolerance: Acceptable range around target
    
    Returns:
        Array of quality scores (0.0 to 1.0)
    """
    n = len(rsi)
    quality = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        distance = abs(rsi[i] - target)
        if distance <= tolerance:
            # Linear scoring: closest to target = 1.0, at edge = 0.5
            quality[i] = 1.0 - (distance / tolerance) * 0.5
        else:
            quality[i] = 0.0
    
    return quality


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Pullback Strategy with RSI Mean Reversion.
    
    Signal Logic:
    1. Identify primary trend using EMA100
    2. Wait for pullback (RSI moves against trend)
    3. Enter when RSI reaches optimal mean-reversion level
    4. Scale position by volatility (ATR)
    
    Entry Conditions:
    - LONG: Price > EMA100 (uptrend) + RSI 35-45 (pullback)
    - SHORT: Price < EMA100 (downtrend) + RSI 55-65 (rally)
    
    Exit: Signal returns to 0 when conditions no longer met
    
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
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Handle NaN values
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    
    # Ensure valid prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators
    ema_trend = calculate_ema(close, EMA_TREND)
    ema_entry = calculate_ema(close, EMA_ENTRY)
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Calculate RSI quality scores
    rsi_long_quality = calculate_rsi_quality(rsi, RSI_LONG_ENTRY, RSI_LONG_RANGE)
    rsi_short_quality = calculate_rsi_quality(rsi, RSI_SHORT_ENTRY, RSI_SHORT_RANGE)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_TREND,
        EMA_ENTRY,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1
    )
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0 or ema_trend[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Calculate trend strength (price separation from EMA)
        price_ema_ratio = (close[i] - ema_trend[i]) / ema_trend[i]
        
        # Determine trend direction
        trend_bullish = price_ema_ratio > TREND_THRESHOLD
        trend_bearish = price_ema_ratio < -TREND_THRESHOLD
        
        # Skip if trend is unclear (price too close to EMA)
        if not trend_bullish and not trend_bearish:
            signals[i] = 0.0
            continue
        
        # Calculate entry signal
        raw_signal = 0.0
        entry_quality = 0.0
        
        if trend_bullish:
            # Long setup: uptrend + RSI pullback
            if rsi_long_quality[i] >= RSI_STRENGTH_MIN:
                # Base signal from RSI quality
                entry_quality = rsi_long_quality[i]
                
                # Bonus for price near entry EMA (confirms pullback)
                if ema_entry[i] > 0:
                    price_ema_entry_ratio = (close[i] - ema_entry[i]) / ema_entry[i]
                    # Prefer price slightly below entry EMA (true pullback)
                    if -0.02 <= price_ema_entry_ratio <= 0.01:
                        entry_quality *= 1.15
                
                raw_signal = entry_quality
        
        elif trend_bearish:
            # Short setup: downtrend + RSI rally
            if rsi_short_quality[i] >= RSI_STRENGTH_MIN:
                # Base signal from RSI quality
                entry_quality = rsi_short_quality[i]
                
                # Bonus for price near entry EMA (confirms rally)
                if ema_entry[i] > 0:
                    price_ema_entry_ratio = (close[i] - ema_entry[i]) / ema_entry[i]
                    # Prefer price slightly above entry EMA (true rally)
                    if -0.01 <= price_ema_entry_ratio <= 0.02:
                        entry_quality *= 1.15
                
                raw_signal = -entry_quality
        
        # Apply volatility adjustment (reduce size in high volatility)
        atr_pct = atr[i] / close[i]
        vol_factor = 1.0
        if atr_pct > 0:
            # Scale down when volatility is high
            vol_factor = min(1.3, VOLATILITY_TARGET / max(atr_pct, 0.003))
        
        signal = raw_signal * vol_factor
        
        # Apply thresholds
        if abs(signal) < MIN_SIGNAL:
            signal = 0.0
        
        signal = np.clip(signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals