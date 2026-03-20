#!/usr/bin/env python3
"""
strategy.py - Trend Funding Hybrid V14
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simplified trend-following with funding rate mean reversion:
    - Primary signal: EMA crossover trend direction (20/50 EMA)
    - Confirmation: Price above/below 200 EMA for trend validation
    - Filter: Extreme funding rates provide contrarian overlay
    - Entry timing: RSI momentum confirmation
    - Volatility filter: ATR-based position sizing
    
    Improvements over V13:
    - Fixed parameter ordering bug (lookback before i in divergence function)
    - Simplified signal combination logic
    - More conservative leverage (2.0 vs 2.5)
    - Reduced complexity to avoid overfitting
    - Better edge case handling

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

name = "trend_funding_hybrid_v14"
timeframe = "1h"
leverage = 2.0  # More conservative after V13 crash

# EMA configuration for trend detection
EMA_FAST = 20
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration for momentum
RSI_PERIOD = 14
RSI_LONG_THRESHOLD = 45
RSI_SHORT_THRESHOLD = 55

# Funding rate configuration
FUNDING_EXTREME_THRESHOLD = 0.0015  # 0.15% per 8hr = very extreme
FUNDING_MODERATE_THRESHOLD = 0.0005  # 0.05% per 8hr = moderate
FUNDING_LOOKBACK = 100
FUNDING_WEIGHT = 0.40  # Funding influence weight

# Volatility configuration
ATR_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0
VOLATILITY_TARGET = 0.015
VOLATILITY_MIN = 0.002
VOLATILITY_MAX = 0.050

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.50

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.15
MAX_SIGNAL = 0.85
SMOOTHING_FACTOR = 0.35
HYSTERESIS_THRESHOLD = 0.10


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
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


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands using only past data.
    Returns: (upper_band, middle_band, lower_band, bandwidth)
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
    
    middle = np.nan_to_num(middle_series.values, nan=0.0)
    std = np.nan_to_num(std_series.values, nan=0.0)
    
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    bandwidth = (upper - lower) / np.where(middle > 0, middle, 1.0)
    bandwidth = np.nan_to_num(bandwidth, nan=0.0)
    
    return upper, middle, lower, bandwidth


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio vs rolling average.
    Only uses past volume data (no look-ahead).
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    rolling_avg = volume_series.rolling(window=lookback, min_periods=lookback).mean()
    
    volume_ratio = np.nan_to_num(volume_series.values / rolling_avg.values, nan=1.0)
    volume_ratio = np.where(volume_ratio <= 0, 1.0, volume_ratio)
    
    return volume_ratio


def calculate_funding_extremes(funding_rate: np.ndarray, lookback: int = 100) -> tuple:
    """
    Calculate rolling percentile extremes of funding rate.
    Returns: (rolling_90th_percentile, rolling_10th_percentile)
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    rolling_high = np.zeros(n, dtype=np.float64)
    rolling_low = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return rolling_high, rolling_low
    
    funding_series = pd.Series(funding_rate)
    rolling_high_series = funding_series.rolling(window=lookback, min_periods=lookback).quantile(0.90)
    rolling_low_series = funding_series.rolling(window=lookback, min_periods=lookback).quantile(0.10)
    
    rolling_high = np.nan_to_num(rolling_high_series.values, nan=0.0)
    rolling_low = np.nan_to_num(rolling_low_series.values, nan=0.0)
    
    return rolling_high, rolling_low


def calculate_funding_signal(funding_rate: np.ndarray, 
                             funding_high: np.ndarray,
                             funding_low: np.ndarray,
                             extreme_threshold: float = 0.0015,
                             moderate_threshold: float = 0.0005,
                             weight: float = 0.40) -> np.ndarray:
    """
    Calculate funding rate contrarian signal.
    Extreme positive funding → short bias (negative signal)
    Extreme negative funding → long bias (positive signal)
    Returns value in [-weight, weight].
    Only uses current/past funding rate (no look-ahead).
    """
    n = len(funding_rate)
    signal = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        fr = funding_rate[i]
        fr_high = funding_high[i]
        fr_low = funding_low[i]
        
        # Determine if funding is extreme based on recent history
        is_extreme_high = fr > extreme_threshold or (fr_high > 0 and fr >= fr_high * 0.85)
        is_extreme_low = fr < -extreme_threshold or (fr_low < 0 and fr <= fr_low * 0.85)
        is_moderate_high = fr > moderate_threshold and not is_extreme_high
        is_moderate_low = fr < -moderate_threshold and not is_extreme_low
        
        if is_extreme_high:
            signal[i] = -weight * min(1.0, fr / extreme_threshold)
        elif is_extreme_low:
            signal[i] = weight * min(1.0, abs(fr) / extreme_threshold)
        elif is_moderate_high:
            signal[i] = -weight * 0.3 * (fr / moderate_threshold)
        elif is_moderate_low:
            signal[i] = weight * 0.3 * (abs(fr) / moderate_threshold)
        else:
            signal[i] = 0.0
    
    return np.clip(signal, -weight, weight)


def detect_rsi_divergence(close: np.ndarray, rsi: np.ndarray, lookback: int, i: int) -> float:
    """
    Detect RSI divergence at index i using only past data.
    Returns: divergence_score in [-1, 1]
    - Positive: bullish divergence (price lower lows, RSI higher lows)
    - Negative: bearish divergence (price higher highs, RSI lower highs)
    
    FIXED: lookback parameter now comes before i (no default after non-default)
    """
    if i < lookback + 1:
        return 0.0
    
    price_window = close[i-lookback:i+1]
    rsi_window = rsi[i-lookback:i+1]
    
    price_min_idx = np.argmin(price_window)
    rsi_min_idx = np.argmin(rsi_window)
    price_max_idx = np.argmax(price_window)
    rsi_max_idx = np.argmax(rsi_window)
    
    divergence_score = 0.0
    
    # Bullish divergence detection
    if price_min_idx > 0 and rsi_min_idx > 0:
        if price_window[price_min_idx] < price_window[0]:
            if rsi_window[rsi_min_idx] > rsi_window[0]:
                divergence_score = 0.5
    
    # Bearish divergence detection
    if price_max_idx > 0 and rsi_max_idx > 0:
        if price_window[price_max_idx] > price_window[0]:
            if rsi_window[rsi_max_idx] < rsi_window[0]:
                divergence_score = -0.5
    
    return divergence_score


def calculate_trend_signal(close: np.ndarray, 
                           ema_fast: np.ndarray,
                           ema_slow: np.ndarray,
                           ema_major: np.ndarray,
                           rsi: np.ndarray,
                           i: int) -> tuple:
    """
    Calculate trend-following signal at index i using only past data.
    Returns: (signal_value, confidence_score)
    """
    if close[i] <= 0 or ema_major[i] <= 0:
        return 0.0, 0.0
    
    # Primary trend direction from EMA crossover
    ema_diff = (ema_fast[i] - ema_slow[i]) / close[i]
    ema_direction = np.sign(ema_diff)
    
    # Major trend filter (price vs 200 EMA)
    major_filter = np.sign(close[i] - ema_major[i])
    
    # Calculate trend strength
    if ema_direction != major_filter and abs(ema_direction) > 0:
        trend_strength = abs(ema_diff) * 50 * 0.5
        confidence = 0.4
    else:
        trend_strength = abs(ema_diff) * 50
        confidence = 0.65
    
    # RSI momentum confirmation
    rsi_factor = 1.0
    if ema_direction > 0:
        if rsi[i] < RSI_LONG_THRESHOLD:
            rsi_factor = 0.4
        elif rsi[i] > 70:
            rsi_factor = 0.7
    elif ema_direction < 0:
        if rsi[i] > RSI_SHORT_THRESHOLD:
            rsi_factor = 0.4
        elif rsi[i] < 30:
            rsi_factor = 0.7
    
    signal = ema_direction * trend_strength * rsi_factor
    
    return signal, confidence


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Funding Hybrid V14 Strategy.
    
    Signal Logic:
    1. Calculate trend signal from EMA crossover (20/50) with 200 EMA filter
    2. Detect RSI divergence for entry timing enhancement
    3. Calculate funding contrarian signal from extreme funding rates
    4. Combine signals with simplified weighting
    5. Apply volatility normalization
    6. Smooth signals with EMA
    7. Apply hysteresis to reduce whipsaws
    8. Filter by minimum signal magnitude
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, funding_rate, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract price data with error handling
    try:
        close = prices["close"].values.astype(np.float64)
        high = prices["high"].values.astype(np.float64)
        low = prices["low"].values.astype(np.float64)
        volume = prices["volume"].values.astype(np.float64)
        
        try:
            funding_rate = prices["funding_rate"].values.astype(np.float64)
            funding_rate = np.nan_to_num(funding_rate, nan=0.0)
        except (KeyError, TypeError, ValueError):
            funding_rate = np.zeros(n, dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Clean data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Fix invalid prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators (all use only past data)
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    bb_upper, bb_middle, bb_lower, bandwidth = calculate_bollinger_bands(
        close, BOLLINGER_PERIOD, BOLLINGER_STD
    )
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    funding_high, funding_low = calculate_funding_extremes(funding_rate, FUNDING_LOOKBACK)
    funding_signal = calculate_funding_signal(
        funding_rate, funding_high, funding_low,
        FUNDING_EXTREME_THRESHOLD, FUNDING_MODERATE_THRESHOLD, FUNDING_WEIGHT
    )
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        FUNDING_LOOKBACK,
        BOLLINGER_PERIOD + 5
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volatility filter (not too low, not too high)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volume filter (ensure sufficient liquidity)
        if volume_ratio[i] < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Calculate trend signal with confidence
        trend_sig, trend_confidence = calculate_trend_signal(
            close, ema_fast, ema_slow, ema_major, rsi, i
        )
        
        # Detect RSI divergence
        rsi_divergence = detect_rsi_divergence(close, rsi, 5, i)
        
        # Get funding overlay
        fund_sig = funding_signal[i]
        
        # Combine signals with simplified weighting
        if abs(trend_sig) > 0.15:
            # Trend is primary driver
            if np.sign(trend_sig) != np.sign(fund_sig) and abs(fund_sig) > 0.10:
                # Conflict between trend and funding - reduce signal
                raw_signal = trend_sig * 0.65 + fund_sig * 0.35
                confidence = trend_confidence * 0.7
            else:
                # Aligned or weak funding - trend dominant
                raw_signal = trend_sig * 0.70 + fund_sig * 0.30
                confidence = trend_confidence
        else:
            # Weak trend - funding can dominate
            raw_signal = fund_sig * 0.55
            confidence = 0.35
        
        # Apply RSI divergence enhancement
        if abs(rsi_divergence) > 0.3 and np.sign(rsi_divergence) == np.sign(raw_signal):
            raw_signal *= 1.10
            confidence = min(1.0, confidence + 0.10)
        elif abs(rsi_divergence) > 0.3 and np.sign(rsi_divergence) != np.sign(raw_signal):
            raw_signal *= 0.75
            confidence = max(0.2, confidence - 0.15)
        
        # Confidence-based scaling
        raw_signal *= (0.65 + 0.35 * confidence)
        
        # Volatility normalization
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.7, 1.5)
        raw_signal *= vol_factor
        
        # Signal smoothing
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Hysteresis: don't flip direction on small changes
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < HYSTERESIS_THRESHOLD:
                smoothed_signal = prev_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals