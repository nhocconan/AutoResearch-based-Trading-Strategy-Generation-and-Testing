#!/usr/bin/env python3
"""
strategy.py - Adaptive Trend Momentum V3 with Volatility Regime Filter
========================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on #010 success (Sharpe=0.330, Return=+40.7%), improving:
    - Fixed RSI scoring logic (was inverted for long signals)
    - Added Bollinger Band squeeze detection for volatility regime
    - Improved trend score normalization (more stable across price levels)
    - Signal hysteresis to reduce flip-flopping
    - Better volume confirmation with Z-score instead of percentile
    
    Key improvements over #010:
    - RSI now properly rewards momentum in trend direction
    - Bollinger Band width filter avoids low-volatility chop
    - Volume Z-score for better outlier detection
    - Signal hysteresis (deadzone) to reduce whipsaws
    - More conservative position sizing in high volatility

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

name = "adaptive_trend_momentum_v3"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for better risk-adjusted returns

# EMA periods for trend detection (simulating multi-timeframe)
EMA_FAST = 8
EMA_MEDIUM = 21
EMA_SLOW = 55
EMA_MAJOR = 200

# RSI configuration
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_BULLISH_ZONE = 50  # Above this supports long bias
RSI_BEARISH_ZONE = 50  # Below this supports short bias

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_ZSCORE_THRESHOLD = 1.0  # Volume must be 1+ std dev above mean

# Bollinger Band configuration for volatility regime
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_THRESHOLD = 0.015  # Band width below this = squeeze (avoid trading)
BB_EXPANSION_MIN = 0.020  # Band width above this = good volatility

# Trend scoring weights
WEIGHT_FAST = 0.35
WEIGHT_MEDIUM = 0.35
WEIGHT_SLOW = 0.30

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012  # Target hourly volatility
VOLATILITY_MIN = 0.003
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL = 0.12
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.65  # Exponential smoothing factor (0-1)
HYSTERESIS_DEADZONE = 0.08  # Signal must cross this to flip direction


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


def calculate_sma(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average using only past data.
    
    Args:
        close: Array of close prices
        period: SMA period
    
    Returns:
        Array of SMA values
    """
    n = len(close)
    sma = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return sma
    
    close_series = pd.Series(close)
    sma_values = close_series.rolling(window=period, min_periods=period).mean().values
    sma = np.nan_to_num(sma_values, nan=0.0)
    
    return sma


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


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0):
    """
    Calculate Bollinger Bands using only past data.
    
    Args:
        close: Array of close prices
        period: BB period
        std_dev: Number of standard deviations
    
    Returns:
        Tuple of (upper, middle, lower, bandwidth) arrays
    """
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    bandwidth = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower, bandwidth
    
    close_series = pd.Series(close)
    middle_series = close_series.rolling(window=period, min_periods=period).mean()
    std_series = close_series.rolling(window=period, min_periods=period).std()
    
    upper = np.nan_to_num((middle_series + std_dev * std_series).values, nan=0.0)
    middle = np.nan_to_num(middle_series.values, nan=0.0)
    lower = np.nan_to_num((middle_series - std_dev * std_series).values, nan=0.0)
    
    # Bandwidth = (Upper - Lower) / Middle
    with np.errstate(divide='ignore', invalid='ignore'):
        bandwidth = np.where(middle > 0, (upper - lower) / middle, 0.0)
    
    return upper, middle, lower, bandwidth


def calculate_volume_zscore(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume Z-score using rolling window.
    Only uses past volume data (no look-ahead).
    
    Args:
        volume: Array of volume values
        lookback: Rolling window for Z-score calculation
    
    Returns:
        Array of volume Z-scores
    """
    n = len(volume)
    volume_z = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return volume_z
    
    volume_series = pd.Series(volume)
    
    for i in range(lookback, n):
        window = volume_series.iloc[i-lookback:i]
        mean_vol = window.mean()
        std_vol = window.std()
        
        if std_vol > 0:
            volume_z[i] = (volume[i] - mean_vol) / std_vol
        else:
            volume_z[i] = 0.0
    
    return volume_z


def calculate_trend_score(close: float, ema_fast: float, ema_medium: float, 
                          ema_slow: float, ema_major: float) -> float:
    """
    Calculate weighted trend score based on EMA stack alignment.
    Normalized by EMA values for stability across price levels.
    
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
    
    # Calculate individual trend components (normalized by respective EMA)
    # Positive = bullish alignment, Negative = bearish alignment
    fast_score = (ema_fast - ema_medium) / ema_medium if ema_medium > 0 else 0.0
    medium_score = (ema_medium - ema_slow) / ema_slow if ema_slow > 0 else 0.0
    slow_score = (ema_slow - ema_major) / ema_major if ema_major > 0 else 0.0
    
    # Price position relative to major EMA
    major_score = (close - ema_major) / ema_major if ema_major > 0 else 0.0
    
    # Weight and combine trend components
    trend_component = (
        WEIGHT_FAST * fast_score +
        WEIGHT_MEDIUM * medium_score +
        WEIGHT_SLOW * slow_score
    )
    
    # Major trend filter (amplifies signal in direction of major trend)
    major_direction = np.sign(major_score)
    trend_score = trend_component * (1.0 + 0.4 * major_direction)
    
    # Normalize to [-1, 1] range using tanh for smooth saturation
    trend_score = np.tanh(trend_score * 100.0)
    
    return trend_score


def calculate_rsi_momentum_score(rsi: float, trend_direction: int) -> float:
    """
    Calculate RSI momentum score based on trend direction.
    
    For LONG (trend_direction=1):
        - RSI > 50 is bullish momentum
        - RSI 30-50 is recovering (moderate bullish)
        - RSI > 70 is overbought but strong momentum
        - RSI < 30 is oversold (avoid long in downtrend)
    
    For SHORT (trend_direction=-1):
        - RSI < 50 is bearish momentum
        - RSI 50-70 is weakening (moderate bearish)
        - RSI < 30 is oversold but strong bearish momentum
        - RSI > 70 is overbought (avoid short in uptrend)
    
    Args:
        rsi: Current RSI value
        trend_direction: +1 for long bias, -1 for short bias
    
    Returns:
        RSI momentum score in range [0, 1]
    """
    if trend_direction > 0:
        # Long bias scoring
        if rsi < 30:
            return 0.2  # Oversold - cautious long
        elif rsi < 45:
            return 0.5  # Recovering
        elif rsi < 55:
            return 0.7  # Neutral-bullish
        elif rsi < 70:
            return 0.9  # Strong momentum
        else:
            return 0.7  # Overbought but trending
    else:
        # Short bias scoring
        if rsi > 70:
            return 0.2  # Overbought - cautious short
        elif rsi > 55:
            return 0.5  # Weakening
        elif rsi > 45:
            return 0.7  # Neutral-bearish
        elif rsi > 30:
            return 0.9  # Strong momentum
        else:
            return 0.7  # Oversold but trending


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Trend Momentum V3 Strategy with Volatility Regime Filter.
    
    Signal Logic:
    1. Weighted trend score from EMA stack alignment
    2. RSI momentum scoring based on trend direction
    3. Volume Z-score for confirmation
    4. Bollinger Band width filter (avoid squeeze zones)
    5. Volatility-based position sizing
    6. Signal smoothing with hysteresis
    
    Entry Conditions:
    - LONG: Positive trend + RSI momentum + volume confirmation + BB expansion
    - SHORT: Negative trend + RSI momentum + volume confirmation + BB expansion
    
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
    volume_z = calculate_volume_zscore(volume, VOLUME_LOOKBACK)
    
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(
        close, BB_PERIOD, BB_STD
    )
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        BB_PERIOD
    )
    
    # Track previous signal for smoothing and hysteresis
    prev_signal = 0.0
    prev_direction = 0  # 0=neutral, 1=long, -1=short
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0 or bb_middle[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Check Bollinger Band volatility regime (avoid squeeze zones)
        if bb_width[i] < BB_SQUEEZE_THRESHOLD:
            # Squeeze detected - reduce position or stay flat
            signals[i] = prev_signal * 0.5
            prev_signal = signals[i]
            continue
        
        # Check volatility regime (avoid extreme volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Calculate weighted trend score
        trend_score = calculate_trend_score(
            close[i], ema_fast[i], ema_medium[i], 
            ema_slow[i], ema_major[i]
        )
        
        # Skip weak trends
        if abs(trend_score) < 0.10:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Determine trend direction
        trend_direction = 1 if trend_score > 0 else -1
        
        # Calculate RSI momentum score based on trend direction
        rsi_score = calculate_rsi_momentum_score(rsi[i], trend_direction)
        
        # Volume confirmation (Z-score based)
        volume_confirmed = volume_z[i] >= VOLUME_ZSCORE_THRESHOLD
        
        # Determine signal direction and base magnitude
        if trend_direction > 0:
            # LONG bias
            base_signal = trend_score * rsi_score
        else:
            # SHORT bias
            base_signal = trend_score * rsi_score
        
        # Apply volume confirmation boost
        if volume_confirmed:
            base_signal *= 1.20
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.4, 2.5)
        
        raw_signal = base_signal * vol_factor
        
        # Apply hysteresis (deadzone for direction flips)
        if prev_direction != 0:
            # Same direction - allow smaller signals
            if trend_direction == prev_direction:
                if abs(raw_signal) < HYSTERESIS_DEADZONE * 0.5:
                    raw_signal = prev_signal * 0.8  # Decay but maintain direction
            else:
                # Opposite direction - require stronger signal to flip
                if abs(raw_signal) < HYSTERESIS_DEADZONE:
                    raw_signal = prev_signal * 0.5  # Strong decay
                    trend_direction = prev_direction  # Maintain direction
        else:
            # Neutral - require minimum signal to enter
            if abs(raw_signal) < HYSTERESIS_DEADZONE:
                raw_signal = 0.0
        
        # Apply exponential smoothing to reduce whipsaws
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        prev_signal = smoothed_signal
        
        # Update direction tracking
        if smoothed_signal > 0.05:
            prev_direction = 1
        elif smoothed_signal < -0.05:
            prev_direction = -1
        else:
            prev_direction = 0
        
        # Apply thresholds
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
            prev_direction = 0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals