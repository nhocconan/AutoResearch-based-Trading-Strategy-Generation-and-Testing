# Strategy: multi_factor_trend_following

## Status
ACTIVE - Sharpe=0.000 | Return=+0.0% | DD=0.0%

## Train Results
| Symbol | Sharpe | Return | Max DD | Trades |
|--------|--------|--------|--------|--------|
| BTCUSDT | 0.000 | +0.0% | 0.0% | 0 |
| ETHUSDT | 0.000 | +0.0% | 0.0% | 0 |
| SOLUSDT | 0.000 | +0.0% | 0.0% | 0 |

## Code
```python
#!/usr/bin/env python3
"""
strategy.py - Multi-Factor Trend Following with Mean Reversion Filter
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Combines trend-following with mean-reversion filters for better risk-adjusted returns.
    - Primary signal: EMA crossover for trend direction
    - Confirmation: RSI momentum (not overextended)
    - Filter: Price relative to Bollinger Bands (avoid extremes)
    - Volume: Confirm trend with above-average volume
    - Volatility: ATR-based position sizing to normalize risk
    
    Key improvement over #001:
    - More conservative entry thresholds
    - Stronger trend confirmation required
    - Mean-reversion filter prevents chasing extremes
    - Lower leverage for better risk management

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

name = "multi_factor_trend_following"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for better risk-adjusted returns

# Strategy parameters - more conservative than #001
EMA_FAST = 12                 # Fast EMA for trend signal
EMA_SLOW = 26                 # Slow EMA for trend baseline
EMA_CONFIRM = 50              # Longer EMA for trend confirmation
RSI_PERIOD = 14               # RSI calculation period
RSI_LONG_MIN = 45             # Minimum RSI for long entries (momentum)
RSI_LONG_MAX = 70             # Maximum RSI for long entries (avoid overbought)
RSI_SHORT_MIN = 30            # Minimum RSI for short entries (avoid oversold)
RSI_SHORT_MAX = 55            # Maximum RSI for short entries (momentum)
BB_PERIOD = 20                # Bollinger Bands period
BB_STD = 2.0                  # Bollinger Bands standard deviation
BB_ENTRY_ZONE = 0.5           # Enter when price in middle 50% of bands (not extreme)
VOLUME_LOOKBACK = 20          # Lookback for volume average
VOLUME_THRESHOLD = 1.2        # Volume must be above this ratio of average
ATR_PERIOD = 14               # ATR calculation period
VOLATILITY_TARGET = 0.012     # Target hourly volatility for position sizing
MIN_SIGNAL = 0.20             # Minimum signal magnitude to trade
MAX_SIGNAL = 0.75             # Maximum signal magnitude
TREND_STRENGTH_MIN = 0.003    # Minimum EMA spread ratio for valid trend


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


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands using only past data.
    
    Args:
        close: Array of close prices
        period: BB period
        std_dev: Number of standard deviations
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band, bb_position)
        bb_position: 0=lower, 0.5=middle, 1=upper
    """
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    position = np.full(n, 0.5, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower, position
    
    close_series = pd.Series(close)
    middle_series = close_series.rolling(window=period, min_periods=period).mean()
    std_series = close_series.rolling(window=period, min_periods=period).std()
    
    upper_series = middle_series + (std_dev * std_series)
    lower_series = middle_series - (std_dev * std_series)
    
    # Calculate position within bands (0=lower, 1=upper)
    band_range = upper_series - lower_series
    mask = band_range > 0
    position[mask] = (close[mask] - lower_series[mask]) / band_range[mask]
    position = np.clip(position, 0.0, 1.0)
    
    upper = np.nan_to_num(upper_series.values, nan=0.0)
    middle = np.nan_to_num(middle_series.values, nan=0.0)
    lower = np.nan_to_num(lower_series.values, nan=0.0)
    
    return upper, middle, lower, position


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
    Multi-Factor Trend Following Strategy with Mean Reversion Filter.
    
    Signal Logic:
    1. Primary trend: EMA fast > EMA slow (bullish) or < (bearish)
    2. Trend confirmation: Price above/below EMA confirm
    3. Momentum filter: RSI in optimal range (not overextended)
    4. Mean reversion filter: Price not at BB extremes
    5. Volume confirmation: Above-average volume
    6. Volatility scaling: ATR-based position sizing
    
    Entry Conditions:
    - LONG: EMA_fast > EMA_slow > EMA_confirm + RSI 45-70 + 
            BB position 0.3-0.7 + volume confirmed
    - SHORT: EMA_fast < EMA_slow < EMA_confirm + RSI 30-55 +
             BB position 0.3-0.7 + volume confirmed
    
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
    
    bb_upper, bb_middle, bb_lower, bb_position = calculate_bollinger_bands(
        close, BB_PERIOD, BB_STD
    )
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_CONFIRM,
        RSI_PERIOD + 1,
        BB_PERIOD,
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
        
        # Filter: trend must be strong enough
        if trend_strength < TREND_STRENGTH_MIN:
            signals[i] = 0.0
            continue
        
        # RSI momentum filter
        rsi_long_ok = RSI_LONG_MIN <= rsi[i] <= RSI_LONG_MAX
        rsi_short_ok = RSI_SHORT_MIN <= rsi[i] <= RSI_SHORT_MAX
        
        # BB position filter (avoid extremes - mean reversion filter)
        bb_position_ok = BB_ENTRY_ZONE <= bb_position[i] <= (1.0 - BB_ENTRY_ZONE)
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_THRESHOLD
        
        # Calculate signal components
        raw_signal = 0.0
        signal_confidence = 0.0
        
        if trend_bullish and rsi_long_ok and bb_position_ok:
            # Long signal
            base_confidence = 0.5
            
            # Add confidence for stronger trend
            trend_factor = min(trend_strength / 0.01, 1.0)
            base_confidence += trend_factor * 0.3
            
            # Add confidence for volume
            if volume_confirmed:
                base_confidence *= 1.2
            
            # RSI quality (prefer RSI around 50-60 for fresh momentum)
            rsi_quality = 1.0
            if 50 <= rsi[i] <= 60:
                rsi_quality = 1.0
            elif 45 <= rsi[i] < 50 or 60 < rsi[i] <= 70:
                rsi_quality = 0.85
            
            signal_confidence = base_confidence * rsi_quality
            raw_signal = signal_confidence
            
        elif trend_bearish and rsi_short_ok and bb_position_ok:
            # Short signal
            base_confidence = 0.5
            
            trend_factor = min(trend_strength / 0.01, 1.0)
            base_confidence += trend_factor * 0.3
            
            if volume_confirmed:
                base_confidence *= 1.2
            
            # RSI quality (prefer RSI around 40-50 for fresh momentum)
            rsi_quality = 1.0
            if 40 <= rsi[i] <= 50:
                rsi_quality = 1.0
            elif 30 <= rsi[i] < 40 or 50 < rsi[i] <= 55:
                rsi_quality = 0.85
            
            signal_confidence = base_confidence * rsi_quality
            raw_signal = -signal_confidence
        
        # Apply volatility adjustment
        atr_pct = atr[i] / close[i]
        vol_factor = 1.0
        if atr_pct > 0:
            vol_factor = min(1.5, VOLATILITY_TARGET / max(atr_pct, 0.001))
        
        signal = raw_signal * vol_factor
        
        # Apply thresholds
        if abs(signal) < MIN_SIGNAL:
            signal = 0.0
        
        signal = np.clip(signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals
```

## Last Updated
2026-03-20 19:57
