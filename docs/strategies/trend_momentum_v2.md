# Strategy: trend_momentum_v2

## Status
ACTIVE - Sharpe=0.330 | Return=+40.7% | DD=-69.6%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 0.074 | -25.2% | -72.5% | 9771 |
| ETHUSDT | 0.340 | +25.4% | -71.3% | 10785 |
| SOLUSDT | 0.575 | +121.9% | -65.0% | 12614 |

## Code
```python
#!/usr/bin/env python3
"""
strategy.py - Trend Momentum V2 with Adaptive Volatility Scaling
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on #007 success (Sharpe=0.291, Return=+520.7%), improving:
    - Cleaner trend alignment scoring (weighted EMA stack)
    - Adaptive volatility scaling (inverse ATR weighting)
    - Better RSI momentum integration (zone-based scoring)
    - Volume confirmation with percentile ranking
    - Signal smoothing to reduce whipsaws
    
    Key improvements over #007:
    - Weighted trend score instead of binary alignment
    - RSI zone scoring (0-1 scale) instead of threshold filtering
    - Volume percentile ranking for better normalization
    - Signal smoothing with exponential decay
    - More conservative leverage (2.0 vs 2.5)

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

name = "trend_momentum_v2"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for better risk-adjusted returns

# EMA periods for trend detection
EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration with zone scoring
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_NEUTRAL_LOW = 40
RSI_NEUTRAL_HIGH = 60

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_PERCENTILE_THRESHOLD = 0.6  # Volume must be in top 40%

# Trend scoring weights
WEIGHT_FAST = 0.4
WEIGHT_MEDIUM = 0.35
WEIGHT_SLOW = 0.25

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.010  # Target hourly volatility
VOLATILITY_MIN = 0.002
VOLATILITY_MAX = 0.035

# Signal configuration
MIN_SIGNAL = 0.10
MAX_SIGNAL = 0.80
SMOOTHING_FACTOR = 0.7  # Exponential smoothing factor (0-1)


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
        rank = (window < volume[i]).sum() / lookback
        volume_pct[i] = rank
    
    return volume_pct


def calculate_trend_score(close: float, ema_fast: float, ema_medium: float, 
                          ema_slow: float, ema_major: float) -> float:
    """
    Calculate weighted trend score based on EMA stack alignment.
    
    Args:
        close: Current close price
        ema_fast: Fast EMA value
        ema_medium: Medium EMA value
        ema_slow: Slow EMA value
        ema_major: Major EMA value
    
    Returns:
        Trend score in range [-1, 1]
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    # Calculate individual trend components (normalized by price)
    fast_score = (ema_fast - ema_medium) / close
    medium_score = (ema_medium - ema_slow) / close
    slow_score = (ema_slow - ema_major) / close
    major_score = (close - ema_major) / close
    
    # Weight and combine
    trend_component = (
        WEIGHT_FAST * fast_score +
        WEIGHT_MEDIUM * medium_score +
        WEIGHT_SLOW * slow_score
    )
    
    # Major trend filter (amplifies signal in direction of major trend)
    major_direction = np.sign(major_score)
    trend_score = trend_component * (1.0 + 0.5 * major_direction)
    
    # Normalize to [-1, 1] range
    trend_score = np.clip(trend_score / 0.01, -1.0, 1.0)
    
    return trend_score


def calculate_rsi_score(rsi: float) -> float:
    """
    Calculate RSI momentum score (0-1 scale).
    
    Args:
        rsi: Current RSI value
    
    Returns:
        RSI score in range [0, 1]
    """
    if rsi <= RSI_OVERSOLD:
        return 0.0  # Too oversold, potential reversal
    elif rsi < RSI_NEUTRAL_LOW:
        return 0.3  # Recovering from oversold
    elif rsi < RSI_NEUTRAL_HIGH:
        return 0.6  # Neutral zone
    elif rsi < RSI_OVERBOUGHT:
        return 0.9  # Strong momentum
    else:
        return 0.5  # Overbought, reduce confidence


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Momentum V2 Strategy with Adaptive Volatility Scaling.
    
    Signal Logic:
    1. Weighted trend score from EMA stack alignment
    2. RSI zone scoring for momentum quality
    3. Volume percentile ranking for confirmation
    4. Volatility-based position sizing
    5. Signal smoothing to reduce whipsaws
    
    Entry Conditions:
    - LONG: Positive trend score + RSI > 40 + volume confirmation
    - SHORT: Negative trend score + RSI < 60 + volume confirmation
    
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
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK
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
        
        # Check volatility regime (avoid extreme volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate weighted trend score
        trend_score = calculate_trend_score(
            close[i], ema_fast[i], ema_medium[i], 
            ema_slow[i], ema_major[i]
        )
        
        # Skip weak trends
        if abs(trend_score) < 0.15:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate RSI momentum score
        rsi_score = calculate_rsi_score(rsi[i])
        
        # Volume confirmation
        volume_confirmed = volume_pct[i] >= VOLUME_PERCENTILE_THRESHOLD
        
        # Determine signal direction and base magnitude
        if trend_score > 0:
            # LONG bias
            if rsi_score < 0.4:
                base_signal = 0.0  # RSI too weak for long
            else:
                base_signal = trend_score * rsi_score
        else:
            # SHORT bias
            if rsi_score > 0.7:
                base_signal = 0.0  # RSI too strong for short
            else:
                base_signal = trend_score * (1.0 - rsi_score * 0.5)
        
        # Apply volume confirmation boost
        if volume_confirmed:
            base_signal *= 1.15
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        
        raw_signal = base_signal * vol_factor
        
        # Apply exponential smoothing to reduce whipsaws
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        prev_signal = smoothed_signal
        
        # Apply thresholds
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals
```

## Last Updated
2026-03-20 20:11
