#!/usr/bin/env python3
"""
strategy.py - Enhanced Multi-Timeframe Trend with Squeeze & Divergence
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on #007 success (Sharpe=0.291, Return=+520.7%), adding:
    - Bollinger Band squeeze detection for breakout timing
    - RSI divergence detection for early trend reversal signals
    - Adaptive EMA periods based on volatility regime
    - Signal smoothing to reduce whipsaws
    - Better volatility-adjusted position sizing
    
    Key improvements over #007:
    - BB squeeze filter (wait for compression before expansion)
    - RSI divergence (2-bar lookback, no look-ahead)
    - Volatility-adaptive EMA periods
    - Signal smoothing with 3-bar EMA
    - Improved trend strength calculation

Look-Ahead Safety:
    - All rolling calculations use only past data (min_periods respected)
    - No .shift(-n) or future index access
    - Signal at bar t uses only prices.iloc[:t+1]
    - RSI divergence uses only past 2 bars (i-2, i-1, i)
"""

import numpy as np
import pandas as pd

# =============================================================================
# Strategy Configuration
# =============================================================================

name = "enhanced_trend_squeeze_divergence"
timeframe = "1h"
leverage = 2.5  # Conservative leverage given crypto volatility

# EMA periods for multi-timeframe trend detection
EMA_FAST_BASE = 9                   # Short-term momentum
EMA_MEDIUM_BASE = 21                # Medium-term trend
EMA_SLOW_BASE = 50                  # Long-term trend
EMA_MAJOR = 200                     # Major trend filter

# Volatility adaptation for EMA periods
VOL_ADAPTATION_FACTOR = 0.5         # How much to adjust EMA periods by vol

# Bollinger Band configuration
BB_PERIOD = 20
BB_STD_DEV = 2.0
BB_SQUEEZE_THRESHOLD = 0.015        # BB width below this = squeeze
BB_EXPANSION_THRESHOLD = 0.025      # BB width above this = expansion

# RSI configuration with divergence detection
RSI_PERIOD = 14
RSI_BASE_LONG_MIN = 45
RSI_BASE_LONG_MAX = 75
RSI_BASE_SHORT_MIN = 25
RSI_BASE_SHORT_MAX = 55
RSI_DIVERGENCE_LOOKBACK = 2         # Bars to look back for divergence

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_THRESHOLD = 1.5
VOLUME_BASE_THRESHOLD = 1.0

# Trend strength thresholds
TREND_STRENGTH_MIN = 0.001
TREND_ALIGNMENT_MIN = 0.0005

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012
VOLATILITY_MIN = 0.003
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL = 0.10
MAX_SIGNAL = 0.85
SIGNAL_SMOOTHING_PERIOD = 3         # EMA period for signal smoothing


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


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands using only past data.
    
    Args:
        close: Array of close prices
        period: BB period
        std_dev: Standard deviation multiplier
    
    Returns:
        Tuple of (upper, middle, lower, width) arrays
    """
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    width = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower, width
    
    close_series = pd.Series(close)
    middle_series = close_series.rolling(window=period, min_periods=period).mean()
    std_series = close_series.rolling(window=period, min_periods=period).std()
    
    middle = np.nan_to_num(middle_series.values, nan=0.0)
    std = np.nan_to_num(std_series.values, nan=0.0)
    
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    width = (upper - lower) / np.where(middle > 0, middle, 1.0)
    
    return upper, middle, lower, width


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


def detect_rsi_divergence(close: np.ndarray, rsi: np.ndarray, lookback: int = 2) -> np.ndarray:
    """
    Detect RSI divergence using only past data.
    
    Bullish divergence: Price makes lower low, RSI makes higher low
    Bearish divergence: Price makes higher high, RSI makes lower high
    
    Args:
        close: Array of close prices
        rsi: Array of RSI values
        lookback: Number of bars to look back for divergence
    
    Returns:
        Array of divergence signals (1=bullish, -1=bearish, 0=none)
    """
    n = len(close)
    divergence = np.zeros(n, dtype=np.float64)
    
    if n < lookback + 2:
        return divergence
    
    for i in range(lookback + 1, n):
        # Get prices and RSI for lookback period
        closes = close[i-lookback-1:i+1]
        rsis = rsi[i-lookback-1:i+1]
        
        # Check for bullish divergence (price lower low, RSI higher low)
        if closes[-1] < closes[0] and rsis[-1] > rsis[0]:
            # Verify it's actually a local minimum
            if closes[-1] <= min(closes[1:-1]) if len(closes) > 2 else True:
                divergence[i] = 1.0
        
        # Check for bearish divergence (price higher high, RSI lower high)
        elif closes[-1] > closes[0] and rsis[-1] < rsis[0]:
            # Verify it's actually a local maximum
            if closes[-1] >= max(closes[1:-1]) if len(closes) > 2 else True:
                divergence[i] = -1.0
    
    return divergence


def calculate_volatility_regime(atr: np.ndarray, close: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calculate volatility regime (0=low, 1=normal, 2=high).
    Uses rolling percentile of ATR/close ratio.
    
    Args:
        atr: Array of ATR values
        close: Array of close prices
        lookback: Rolling window for regime calculation
    
    Returns:
        Array of volatility regime values (0, 1, or 2)
    """
    n = len(close)
    regime = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return regime
    
    atr_pct = atr / np.where(close > 0, close, 1.0)
    atr_pct_series = pd.Series(atr_pct)
    
    for i in range(lookback, n):
        window = atr_pct_series.iloc[i-lookback:i]
        percentile = (window <= atr_pct[i]).mean()
        
        if percentile < 0.3:
            regime[i] = 0.0  # Low volatility
        elif percentile < 0.7:
            regime[i] = 1.0  # Normal volatility
        else:
            regime[i] = 2.0  # High volatility
    
    return regime


def smooth_signal(signals: np.ndarray, period: int = 3) -> np.ndarray:
    """
    Smooth signals using EMA to reduce whipsaws.
    Only uses past signal data (no look-ahead).
    
    Args:
        signals: Array of raw signals
        period: Smoothing period
    
    Returns:
        Array of smoothed signals
    """
    n = len(signals)
    smoothed = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return signals.copy()
    
    signal_series = pd.Series(signals)
    smoothed_series = signal_series.ewm(span=period, adjust=False, min_periods=period).mean()
    smoothed = np.nan_to_num(smoothed_series.values, nan=0.0)
    
    return smoothed


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Enhanced Multi-Timeframe Trend Strategy with Squeeze & Divergence.
    
    Signal Logic:
    1. Major trend: Price relative to 200 EMA defines overall direction
    2. Trend alignment: EMA stack alignment (9/21/50/200)
    3. Bollinger squeeze: Wait for compression before expansion
    4. RSI divergence: Early reversal signals
    5. Momentum filter: RSI in reasonable range
    6. Volume confirmation: Volume spike for breakout validation
    7. Volatility scaling: ATR-based position sizing with regime adjustment
    
    Entry Conditions:
    - LONG: Price > EMA200 + EMA stack bullish + BB expanding + (RSI ok OR bullish divergence)
    - SHORT: Price < EMA200 + EMA stack bearish + BB expanding + (RSI ok OR bearish divergence)
    
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
    
    # Calculate volatility for adaptive EMA periods
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    atr_pct = atr / np.where(close > 0, close, 1.0)
    
    # Adaptive EMA periods based on volatility
    vol_factor = np.clip(atr_pct / VOLATILITY_TARGET, 0.5, 2.0)
    ema_fast = int(EMA_FAST_BASE * vol_factor)
    ema_medium = int(EMA_MEDIUM_BASE * vol_factor)
    ema_slow = int(EMA_SLOW_BASE * vol_factor)
    
    # Ensure minimum periods
    ema_fast = max(ema_fast, 5)
    ema_medium = max(ema_medium, 10)
    ema_slow = max(ema_slow, 20)
    
    # Calculate all indicators
    ema_fast_arr = calculate_ema(close, ema_fast)
    ema_medium_arr = calculate_ema(close, ema_medium)
    ema_slow_arr = calculate_ema(close, ema_slow)
    ema_major_arr = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(
        close, BB_PERIOD, BB_STD_DEV
    )
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    vol_regime = calculate_volatility_regime(atr, close, 50)
    rsi_divergence = detect_rsi_divergence(close, rsi, RSI_DIVERGENCE_LOOKBACK)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        ema_slow + 10,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        BB_PERIOD,
        50  # For volatility regime
    )
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Check volatility regime (avoid extreme volatility)
        current_atr_pct = atr[i] / close[i]
        if current_atr_pct < VOLATILITY_MIN or current_atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            continue
        
        # Major trend filter (200 EMA)
        price_vs_major = (close[i] - ema_major_arr[i]) / close[i]
        major_trend_bullish = price_vs_major > TREND_ALIGNMENT_MIN
        major_trend_bearish = price_vs_major < -TREND_ALIGNMENT_MIN
        
        # EMA stack alignment
        ema_stack_bullish = (
            ema_fast_arr[i] > ema_medium_arr[i] and
            ema_medium_arr[i] > ema_slow_arr[i] and
            ema_slow_arr[i] > ema_major_arr[i]
        )
        ema_stack_bearish = (
            ema_fast_arr[i] < ema_medium_arr[i] and
            ema_medium_arr[i] < ema_slow_arr[i] and
            ema_slow_arr[i] < ema_major_arr[i]
        )
        
        # Calculate trend strength
        trend_strength_fast = abs(ema_fast_arr[i] - ema_medium_arr[i]) / close[i]
        trend_strength_medium = abs(ema_medium_arr[i] - ema_slow_arr[i]) / close[i]
        trend_strength_slow = abs(ema_slow_arr[i] - ema_major_arr[i]) / close[i]
        trend_strength = min(trend_strength_fast, trend_strength_medium, trend_strength_slow)
        
        # Filter: trend must be strong enough
        if trend_strength < TREND_STRENGTH_MIN:
            signals[i] = 0.0
            continue
        
        # Bollinger Band squeeze/expansion filter
        bb_expanding = bb_width[i] > BB_SQUEEZE_THRESHOLD
        bb_not_extreme = bb_width[i] < BB_EXPANSION_THRESHOLD * 2
        
        # Dynamic RSI thresholds based on trend strength
        rsi_adjustment = min(trend_strength / 0.005, 0.5)
        rsi_long_min = RSI_BASE_LONG_MIN - rsi_adjustment * 10
        rsi_long_max = RSI_BASE_LONG_MAX + rsi_adjustment * 5
        rsi_short_min = RSI_BASE_SHORT_MIN - rsi_adjustment * 5
        rsi_short_max = RSI_BASE_SHORT_MAX + rsi_adjustment * 10
        
        # RSI momentum filter
        rsi_long_ok = rsi_long_min <= rsi[i] <= rsi_long_max
        rsi_short_ok = rsi_short_min <= rsi[i] <= rsi_short_max
        
        # RSI divergence signals
        bullish_div = rsi_divergence[i] == 1.0
        bearish_div = rsi_divergence[i] == -1.0
        
        # Volume confirmation
        volume_base_ok = volume_ratio[i] >= VOLUME_BASE_THRESHOLD
        volume_spike = volume_ratio[i] >= VOLUME_SPIKE_THRESHOLD
        
        # Calculate signal
        raw_signal = 0.0
        signal_confidence = 0.0
        
        # LONG signal
        if major_trend_bullish and ema_stack_bullish and bb_expanding and bb_not_extreme:
            # Need either RSI ok OR bullish divergence
            if rsi_long_ok or bullish_div:
                base_confidence = 0.5
                
                # Trend strength factor
                trend_factor = min(trend_strength / 0.006, 1.0)
                base_confidence += trend_factor * 0.3
                
                # Divergence boost
                if bullish_div:
                    base_confidence += 0.15
                
                # Volume boost
                if volume_spike:
                    base_confidence *= 1.2
                elif volume_base_ok:
                    base_confidence *= 1.05
                
                # RSI quality (prefer momentum in trending market)
                rsi_quality = 1.0
                if 50 <= rsi[i] <= 65:
                    rsi_quality = 1.0
                elif rsi_long_min <= rsi[i] < 50 or 65 < rsi[i] <= rsi_long_max:
                    rsi_quality = 0.85
                
                # Volatility regime adjustment
                regime_factor = 1.0
                if vol_regime[i] == 0:
                    regime_factor = 1.1  # Low vol = more confidence
                elif vol_regime[i] == 2:
                    regime_factor = 0.8  # High vol = less confidence
                
                signal_confidence = base_confidence * rsi_quality * regime_factor
                raw_signal = signal_confidence
        
        # SHORT signal
        elif major_trend_bearish and ema_stack_bearish and bb_expanding and bb_not_extreme:
            # Need either RSI ok OR bearish divergence
            if rsi_short_ok or bearish_div:
                base_confidence = 0.5
                
                trend_factor = min(trend_strength / 0.006, 1.0)
                base_confidence += trend_factor * 0.3
                
                # Divergence boost
                if bearish_div:
                    base_confidence += 0.15
                
                if volume_spike:
                    base_confidence *= 1.2
                elif volume_base_ok:
                    base_confidence *= 1.05
                
                # RSI quality
                rsi_quality = 1.0
                if 35 <= rsi[i] <= 50:
                    rsi_quality = 1.0
                elif rsi_short_min <= rsi[i] < 35 or 50 < rsi[i] <= rsi_short_max:
                    rsi_quality = 0.85
                
                # Volatility regime adjustment
                regime_factor = 1.0
                if vol_regime[i] == 0:
                    regime_factor = 1.1
                elif vol_regime[i] == 2:
                    regime_factor = 0.8
                
                signal_confidence = base_confidence * rsi_quality * regime_factor
                raw_signal = -signal_confidence
        
        # Apply volatility adjustment for position sizing
        vol_factor = 1.0
        if current_atr_pct > 0:
            vol_factor = min(1.5, VOLATILITY_TARGET / max(current_atr_pct, 0.001))
        
        signal = raw_signal * vol_factor
        
        # Apply thresholds
        if abs(signal) < MIN_SIGNAL:
            signal = 0.0
        
        signal = np.clip(signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    # Smooth signals to reduce whipsaws
    signals = smooth_signal(signals, SIGNAL_SMOOTHING_PERIOD)
    
    # Final clip to ensure valid range
    signals = np.clip(signals, -1.0, 1.0)
    
    return signals