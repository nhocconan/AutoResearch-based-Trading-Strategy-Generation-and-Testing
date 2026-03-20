# Strategy: trend_momentum_rsi_filter

## Status
ACTIVE - Sharpe=-0.990 | Return=-81.9% | DD=-96.1%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -2.620 | -99.8% | -99.8% | 4889 |
| ETHUSDT | -0.595 | -91.6% | -96.4% | 4889 |
| SOLUSDT | 0.245 | -54.2% | -91.9% | 5291 |

## Code
```python
#!/usr/bin/env python3
"""
strategy.py - Trend Momentum with RSI Filter and Volatility Scaling
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Trend-following with momentum confirmation and mean-reversion filter.
    - Use 50/200 EMA for long-term trend direction
    - RSI(14) filters entries to avoid overbought/oversold extremes
    - Volume confirmation for breakout validity
    - ATR-based volatility scaling for position sizing
    - Conservative leverage to account for crypto volatility

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

name = "trend_momentum_rsi_filter"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for crypto volatility

# Strategy parameters
EMA_TREND_FAST = 50           # Fast EMA for trend direction
EMA_TREND_SLOW = 200          # Slow EMA for long-term trend
RSI_PERIOD = 14               # RSI calculation period
RSI_OVERBOUGHT = 70           # RSI overbought threshold
RSI_OVERSOLD = 30             # RSI oversold threshold
VOLUME_LOOKBACK = 20          # Lookback for volume average
VOLUME_THRESHOLD = 1.3        # Volume spike multiplier
ATR_PERIOD = 14               # ATR calculation period
VOLATILITY_TARGET = 0.02      # Target volatility for position sizing
MIN_SIGNAL = 0.2              # Minimum signal magnitude to trade
MAX_SIGNAL = 0.8              # Maximum signal magnitude (leave room for scaling)


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
    
    # Initialize with SMA
    ema[period - 1] = np.mean(close[:period])
    
    # Calculate EMA multiplier
    multiplier = 2.0 / (period + 1)
    
    # Calculate EMA for remaining periods
    for i in range(period, n):
        ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    
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
    rsi = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    delta = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        delta[i] = close[i] - close[i-1]
    
    # Separate gains and losses
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        if delta[i] > 0:
            gains[i] = delta[i]
        else:
            losses[i] = -delta[i]
    
    # Calculate initial average gain/loss using SMA
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    
    rsi[period] = 50.0  # Default if no data
    
    if avg_loss != 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate RSI using Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
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


def calculate_price_momentum(close: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Calculate price momentum as percentage change over period.
    Only uses past data.
    
    Args:
        close: Array of close prices
        period: Momentum lookback period
    
    Returns:
        Array of momentum values (percentage)
    """
    n = len(close)
    momentum = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return momentum
    
    for i in range(period, n):
        if close[i-period] > 0:
            momentum[i] = (close[i] - close[i-period]) / close[i-period]
    
    return momentum


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Momentum Strategy with RSI Filter and Volatility Scaling.
    
    Signal Logic:
    1. Calculate 50/200 EMA for long-term trend direction
    2. Calculate RSI(14) to filter overbought/oversold conditions
    3. Calculate ATR for volatility-based position sizing
    4. Calculate volume ratio for confirmation
    5. Calculate price momentum for entry timing
    
    Entry Conditions:
    - LONG: Fast EMA > Slow EMA AND RSI < 70 AND volume confirmed AND momentum positive
    - SHORT: Fast EMA < Slow EMA AND RSI > 30 AND volume confirmed AND momentum negative
    
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
    
    # Handle any NaN values in price data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Ensure no zero or negative prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate EMAs for trend direction
    ema_fast = calculate_ema(close, EMA_TREND_FAST)
    ema_slow = calculate_ema(close, EMA_TREND_SLOW)
    
    # Calculate RSI for entry filtering
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Calculate ATR for volatility adjustment
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Calculate volume ratio
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Calculate price momentum
    momentum = calculate_price_momentum(close, period=10)
    
    # Calculate trend strength (EMA spread normalized by price)
    ema_spread = (ema_fast - ema_slow) / close
    ema_spread = np.nan_to_num(ema_spread, nan=0.0)
    
    # Determine minimum valid index (all indicators need warmup period)
    min_valid_index = max(
        EMA_TREND_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        10  # momentum period
    )
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip if any required data is invalid
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Trend direction from EMA relationship
        trend_bullish = ema_fast[i] > ema_slow[i]
        trend_bearish = ema_fast[i] < ema_slow[i]
        
        # RSI filter - avoid buying overbought, selling oversold
        rsi_neutral = (RSI_OVERSOLD < rsi[i] < RSI_OVERBOUGHT)
        rsi_bullish_ok = rsi[i] < RSI_OVERBOUGHT
        rsi_bearish_ok = rsi[i] > RSI_OVERSOLD
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_THRESHOLD
        
        # Momentum confirmation
        momentum_positive = momentum[i] > 0
        momentum_negative = momentum[i] < 0
        
        # Calculate trend strength (normalized)
        trend_strength = min(abs(ema_spread[i]) * 50, 1.0)  # Cap at 1.0
        
        # Volatility adjustment (reduce position in high volatility)
        atr_pct = atr[i] / close[i]
        vol_factor = 1.0
        if atr_pct > 0:
            # Scale inversely to volatility, target ~2% hourly volatility
            vol_factor = min(1.5, VOLATILITY_TARGET / max(atr_pct, 0.001))
        
        # RSI quality factor (better signals when RSI is in middle range)
        rsi_quality = 1.0
        if rsi_neutral:
            # Best quality when RSI is in 40-60 range
            rsi_center = abs(rsi[i] - 50)
            rsi_quality = 1.0 - (rsi_center / 50)
            rsi_quality = max(0.5, rsi_quality)
        else:
            rsi_quality = 0.7  # Reduced quality at extremes
        
        # Base signal from trend direction
        raw_signal = 0.0
        signal_confidence = 0.0
        
        if trend_bullish and rsi_bullish_ok and momentum_positive:
            # Long signal
            signal_confidence = 1.0
            if volume_confirmed:
                signal_confidence += 0.3
            raw_signal = trend_strength * signal_confidence
        elif trend_bearish and rsi_bearish_ok and momentum_negative:
            # Short signal
            signal_confidence = 1.0
            if volume_confirmed:
                signal_confidence += 0.3
            raw_signal = -trend_strength * signal_confidence
        
        # Apply RSI quality factor
        raw_signal *= rsi_quality
        
        # Apply volatility adjustment
        signal = raw_signal * vol_factor
        
        # Apply minimum signal threshold
        if abs(signal) < MIN_SIGNAL:
            signal = 0.0
        
        # Clip to [-MAX_SIGNAL, MAX_SIGNAL] to leave room for portfolio scaling
        signal = np.clip(signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals
```

## Last Updated
2026-03-20 19:53
