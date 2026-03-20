# Strategy: simplified_trend_momentum

## Status
ACTIVE - Sharpe=0.208 | Return=+33.9% | DD=-89.7%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | -0.352 | -91.6% | -97.1% | 3070 |
| ETHUSDT | 0.110 | -75.0% | -94.6% | 4210 |
| SOLUSDT | 0.864 | +268.4% | -77.5% | 7388 |

## Code
```python
#!/usr/bin/env python3
"""
strategy.py - Simplified Trend Following with Momentum Confirmation
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simpler trend-following approach after #002 was too restrictive (Sharpe=0.000).
    - Primary signal: EMA crossover for trend direction
    - Confirmation: RSI momentum (relaxed thresholds)
    - Volume: Secondary confirmation only
    - Removed: BB position filter (was too restrictive)
    
    Key improvement over #002:
    - Fewer filters = more trade opportunities
    - Relaxed RSI thresholds
    - Lower trend strength requirement
    - Simpler signal calculation
    - Slightly higher leverage (2.5 vs 2.0)

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

name = "simplified_trend_momentum"
timeframe = "1h"
leverage = 2.5  # Moderate leverage for better capture of trends

# Strategy parameters - simpler and less restrictive than #002
EMA_FAST = 9                  # Faster EMA for quicker trend detection
EMA_SLOW = 21                 # Slower EMA for trend baseline
EMA_CONFIRM = 50              # Longer EMA for trend confirmation
RSI_PERIOD = 14               # RSI calculation period
RSI_LONG_MIN = 40             # Relaxed minimum RSI for long entries
RSI_LONG_MAX = 75             # Relaxed maximum RSI for long entries
RSI_SHORT_MIN = 25            # Relaxed minimum RSI for short entries
RSI_SHORT_MAX = 60            # Relaxed maximum RSI for short entries
VOLUME_LOOKBACK = 20          # Lookback for volume average
VOLUME_THRESHOLD = 1.0        # Volume threshold (relaxed from 1.2)
ATR_PERIOD = 14               # ATR calculation period
VOLATILITY_TARGET = 0.015     # Target hourly volatility for position sizing
MIN_SIGNAL = 0.15             # Lower minimum signal magnitude to trade
MAX_SIGNAL = 0.80             # Maximum signal magnitude
TREND_STRENGTH_MIN = 0.0015   # Lower minimum EMA spread ratio for valid trend


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
    
    mask = rolling_avg > 0
    volume_ratio[mask] = volume[mask] / rolling_avg[mask]
    
    return volume_ratio


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Simplified Trend Following Strategy with Momentum Confirmation.
    
    Signal Logic:
    1. Primary trend: EMA fast > EMA slow (bullish) or < (bearish)
    2. Trend confirmation: Price above/below EMA confirm
    3. Momentum filter: RSI in reasonable range (relaxed thresholds)
    4. Volume confirmation: Optional boost (not required)
    5. Volatility scaling: ATR-based position sizing
    
    Entry Conditions:
    - LONG: EMA_fast > EMA_slow > EMA_confirm + RSI 40-75
    - SHORT: EMA_fast < EMA_slow < EMA_confirm + RSI 25-60
    
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
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_confirm = calculate_ema(close, EMA_CONFIRM)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_CONFIRM,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK
    )
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Calculate trend strength (EMA spread normalized)
        ema_spread_fast_slow = (ema_fast[i] - ema_slow[i]) / close[i]
        ema_spread_slow_confirm = (ema_slow[i] - ema_confirm[i]) / close[i]
        
        # Trend direction and strength
        trend_bullish = (ema_fast[i] > ema_slow[i] > ema_confirm[i])
        trend_bearish = (ema_fast[i] < ema_slow[i] < ema_confirm[i])
        
        trend_strength = min(abs(ema_spread_fast_slow), abs(ema_spread_slow_confirm))
        
        # Filter: trend must be strong enough (relaxed from #002)
        if trend_strength < TREND_STRENGTH_MIN:
            signals[i] = 0.0
            continue
        
        # RSI momentum filter (relaxed thresholds)
        rsi_long_ok = RSI_LONG_MIN <= rsi[i] <= RSI_LONG_MAX
        rsi_short_ok = RSI_SHORT_MIN <= rsi[i] <= RSI_SHORT_MAX
        
        # Volume confirmation (optional boost, not required)
        volume_confirmed = volume_ratio[i] >= VOLUME_THRESHOLD
        
        # Calculate signal
        raw_signal = 0.0
        signal_confidence = 0.0
        
        if trend_bullish and rsi_long_ok:
            # Long signal
            base_confidence = 0.5
            
            # Add confidence for stronger trend
            trend_factor = min(trend_strength / 0.008, 1.0)
            base_confidence += trend_factor * 0.3
            
            # Volume boost (optional)
            if volume_confirmed:
                base_confidence *= 1.15
            
            # RSI quality (prefer momentum but not overextended)
            rsi_quality = 1.0
            if 50 <= rsi[i] <= 65:
                rsi_quality = 1.0
            elif 40 <= rsi[i] < 50 or 65 < rsi[i] <= 75:
                rsi_quality = 0.9
            
            signal_confidence = base_confidence * rsi_quality
            raw_signal = signal_confidence
            
        elif trend_bearish and rsi_short_ok:
            # Short signal
            base_confidence = 0.5
            
            trend_factor = min(trend_strength / 0.008, 1.0)
            base_confidence += trend_factor * 0.3
            
            if volume_confirmed:
                base_confidence *= 1.15
            
            # RSI quality
            rsi_quality = 1.0
            if 35 <= rsi[i] <= 50:
                rsi_quality = 1.0
            elif 25 <= rsi[i] < 35 or 50 < rsi[i] <= 60:
                rsi_quality = 0.9
            
            signal_confidence = base_confidence * rsi_quality
            raw_signal = -signal_confidence
        
        # Apply volatility adjustment
        atr_pct = atr[i] / close[i]
        vol_factor = 1.0
        if atr_pct > 0:
            vol_factor = min(1.5, VOLATILITY_TARGET / max(atr_pct, 0.001))
        
        signal = raw_signal * vol_factor
        
        # Apply thresholds (relaxed from #002)
        if abs(signal) < MIN_SIGNAL:
            signal = 0.0
        
        signal = np.clip(signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals
```

## Last Updated
2026-03-20 19:59
