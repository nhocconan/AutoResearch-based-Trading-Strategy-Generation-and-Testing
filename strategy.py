#!/usr/bin/env python3
"""
strategy.py - Trend Momentum with Funding Rate Awareness
=======================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simplified trend-following with momentum confirmation and funding rate awareness.
    - Use 12/26 EMA crossover for trend direction (standard MACD periods)
    - ROC momentum confirms trend strength
    - Funding rate filter avoids crowded trades
    - Volume filter is less restrictive to ensure signal generation
    - Lower signal thresholds for more trading opportunities

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

name = "trend_momentum_funding"
timeframe = "1h"
leverage = 2.0  # Conservative leverage after poor historical performance

# Strategy parameters
EMA_FAST = 12             # Fast EMA period (standard MACD)
EMA_SLOW = 26             # Slow EMA period (standard MACD)
EMA_SIGNAL = 9            # Signal line EMA for momentum
ROC_PERIOD = 10           # Rate of Change period for momentum
VOLUME_LOOKBACK = 20      # Lookback for volume average
VOLUME_THRESHOLD = 1.2    # Volume spike multiplier (reduced from 1.5)
ATR_PERIOD = 14           # ATR calculation period
MIN_SIGNAL = 0.15         # Minimum signal magnitude to trade (reduced from 0.3)
FUNDING_THRESHOLD = 0.0005  # Funding rate threshold for filter


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


def calculate_roc(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Rate of Change (momentum) using only past data.
    
    Args:
        close: Array of close prices
        period: ROC period
    
    Returns:
        Array of ROC values (percentage change)
    """
    n = len(close)
    roc = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return roc
    
    for i in range(period, n):
        if close[i - period] > 0:
            roc[i] = (close[i] - close[i - period]) / close[i - period]
    
    return roc


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio relative to rolling average.
    Only uses past volume data (no look-ahead).
    
    Args:
        volume: Array of volume values
        lookback: Rolling window for average calculation
    
    Returns:
        Array of volume ratios
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    rolling_avg = volume_series.rolling(window=lookback, min_periods=lookback).mean().values
    
    # Avoid division by zero
    mask = rolling_avg > 0
    volume_ratio[mask] = volume[mask] / rolling_avg[mask]
    
    return volume_ratio


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Momentum Strategy with Funding Rate Awareness.
    
    Signal Logic:
    1. Calculate fast/slow EMA for trend direction
    2. Calculate ROC for momentum confirmation
    3. Calculate ATR for volatility adjustment
    4. Calculate volume ratio for confirmation
    5. Check funding rate if available (avoid crowded trades)
    6. Generate signals based on trend + momentum + volume
    
    Entry Conditions:
    - LONG: Fast EMA > Slow EMA AND ROC > 0 AND volume confirmed
    - SHORT: Fast EMA < Slow EMA AND ROC < 0 AND volume confirmed
    - Reduce position if funding rate is extreme (crowded trade)
    
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
    except (KeyError, TypeError, ValueError) as e:
        # Return zeros if required columns missing
        return signals
    
    # Try to get funding rate if available (optional column)
    funding_rate = None
    if "funding_rate" in prices.columns:
        try:
            funding_rate = prices["funding_rate"].values.astype(np.float64)
            funding_rate = np.nan_to_num(funding_rate, nan=0.0)
        except:
            funding_rate = None
    
    # Handle any NaN values in price data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Ensure no zero or negative prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate EMAs
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    
    # Calculate ATR for volatility adjustment
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Calculate volume ratio
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Calculate ROC for momentum
    roc = calculate_roc(close, ROC_PERIOD)
    
    # Calculate EMA spread for trend strength
    ema_spread = (ema_fast - ema_slow) / close
    ema_spread = np.nan_to_num(ema_spread, nan=0.0)
    
    # Normalize ROC to similar scale as ema_spread
    roc_normalized = roc * 100  # Convert to percentage-like scale
    roc_normalized = np.nan_to_num(roc_normalized, nan=0.0)
    
    # Determine minimum valid index
    min_valid_index = max(EMA_SLOW, ATR_PERIOD + 1, VOLUME_LOOKBACK, ROC_PERIOD + 1)
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip if any required data is invalid
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Trend direction from EMA crossover
        ema_diff = ema_fast[i] - ema_slow[i]
        ema_bullish = ema_diff > 0
        ema_bearish = ema_diff < 0
        
        # Momentum confirmation
        momentum_bullish = roc[i] > 0
        momentum_bearish = roc[i] < 0
        
        # Volume confirmation (less restrictive)
        volume_confirmed = volume_ratio[i] >= VOLUME_THRESHOLD
        
        # Combine trend and momentum for signal strength
        trend_strength = 0.0
        if ema_bullish and momentum_bullish:
            # Both trend and momentum agree bullish
            trend_strength = min(abs(ema_spread[i]) * 50 + abs(roc_normalized[i]) * 0.5, 1.0)
        elif ema_bearish and momentum_bearish:
            # Both trend and momentum agree bearish
            trend_strength = -min(abs(ema_spread[i]) * 50 + abs(roc_normalized[i]) * 0.5, 1.0)
        else:
            # Trend and momentum disagree - reduce signal
            if ema_bullish:
                trend_strength = min(abs(ema_spread[i]) * 25, 0.5)
            elif ema_bearish:
                trend_strength = -min(abs(ema_spread[i]) * 25, 0.5)
        
        # Apply volume confirmation (reduce but don't eliminate signal)
        if not volume_confirmed:
            trend_strength *= 0.7
        
        # Volatility adjustment (position sizing)
        atr_pct = atr[i] / close[i]
        vol_factor = 1.0
        if atr_pct > 0:
            # Typical 1h ATR% is 0.5-2%, scale inversely but less aggressively
            vol_factor = min(1.0, 0.02 / max(atr_pct, 0.001))
        
        signal = trend_strength * vol_factor
        
        # Funding rate filter (if available)
        if funding_rate is not None:
            # Extreme positive funding → reduce long positions (crowded)
            # Extreme negative funding → reduce short positions (crowded)
            if signal > 0 and funding_rate[i] > FUNDING_THRESHOLD:
                signal *= 0.5  # Reduce long when funding is expensive
            elif signal < 0 and funding_rate[i] < -FUNDING_THRESHOLD:
                signal *= 0.5  # Reduce short when funding is very negative
        
        # Apply minimum signal threshold (lowered for more trades)
        if abs(signal) < MIN_SIGNAL:
            signal = 0.0
        
        # Clip to [-1, 1]
        signal = np.clip(signal, -1.0, 1.0)
        
        signals[i] = signal
    
    return signals