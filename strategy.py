#!/usr/bin/env python3
"""
strategy.py - Volatility Regime Trend V13
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "4h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Trade trends only during normal volatility regimes to avoid whipsaws:
    - Primary signal: EMA crossover (12/26) with 100 EMA trend filter
    - Volatility regime: Only trade when ATR is in middle 50% of recent range
    - Momentum confirmation: RSI must confirm direction (not extreme)
    - Funding overlay: Reduce position size when funding is extreme
    - Signal smoothing: Apply hysteresis to reduce flip-flopping
    
    Why 4h timeframe:
    - Cleaner signals than 1h, less noise than 1d
    - Fewer trades but higher quality
    - Better risk/reward for trend following
    - Funding rates apply every 8h, aligns well with 4h bars
    
    Drawdown Control:
    - Volatility regime filter avoids trading during chaotic periods
    - Stricter entry thresholds reduce false signals
    - Signal smoothing prevents rapid direction changes

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

name = "volatility_regime_trend_v13"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for drawdown control

# EMA configuration for trend detection
EMA_FAST = 12
EMA_SLOW = 26
EMA_MAJOR = 100

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_MIN = 45  # RSI must be above this for longs
RSI_LONG_MAX = 65  # RSI must be below this for longs (not overbought)
RSI_SHORT_MIN = 35  # RSI must be above this for shorts (not oversold)
RSI_SHORT_MAX = 55  # RSI must be below this for shorts

# Volatility regime configuration
ATR_PERIOD = 20
ATR_LOOKBACK = 100  # For calculating volatility percentiles
VOLATILITY_LOWER_PERCENTILE = 0.25  # Only trade above 25th percentile
VOLATILITY_UPPER_PERCENTILE = 0.75  # Only trade below 75th percentile

# Funding rate configuration
FUNDING_EXTREME_THRESHOLD = 0.0008  # 0.08% per 8hr
FUNDING_LOOKBACK = 50
FUNDING_IMPACT = 0.30  # How much funding reduces signal strength

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.25  # Minimum signal to generate position
MAX_SIGNAL = 0.75  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.40  # EMA smoothing factor for signals
HYSTERESIS_THRESHOLD = 0.15  # Minimum change to flip signal direction

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.60


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


def calculate_volatility_regime(atr_pct: np.ndarray, lookback: int = 100,
                                 lower_pct: float = 0.25, upper_pct: float = 0.75) -> np.ndarray:
    """
    Determine if current volatility is in tradable regime.
    Returns 1.0 if in regime, 0.0 if not.
    Only uses past volatility data (no look-ahead).
    """
    n = len(atr_pct)
    regime = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return regime
    
    atr_series = pd.Series(atr_pct)
    
    for i in range(lookback, n):
        window = atr_series.iloc[i-lookback:i]
        lower_bound = window.quantile(lower_pct)
        upper_bound = window.quantile(upper_pct)
        
        if lower_bound <= atr_pct[i] <= upper_bound:
            regime[i] = 1.0
        else:
            regime[i] = 0.0
    
    return regime


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
    
    return volume_ratio


def calculate_funding_impact(funding_rate: np.ndarray, 
                              threshold: float = 0.0008,
                              impact: float = 0.30) -> np.ndarray:
    """
    Calculate funding rate impact on signal strength.
    Extreme funding reduces signal strength (crowded trade warning).
    Returns multiplier in [1-impact, 1.0].
    Only uses current/past funding rate (no look-ahead).
    """
    n = len(funding_rate)
    multiplier = np.ones(n, dtype=np.float64)
    
    for i in range(n):
        fr = abs(funding_rate[i])
        if fr > threshold:
            # Reduce signal strength proportionally to funding extremeness
            excess = min(1.0, (fr - threshold) / threshold)
            multiplier[i] = 1.0 - (impact * excess)
        else:
            multiplier[i] = 1.0
    
    return multiplier


def calculate_trend_signal(close: np.ndarray, 
                           ema_fast: np.ndarray,
                           ema_slow: np.ndarray,
                           ema_major: np.ndarray,
                           rsi: np.ndarray) -> np.ndarray:
    """
    Calculate trend-following signal based on EMA crossover and RSI.
    Returns value in [-1, 1].
    Only uses current/past data (no look-ahead).
    """
    n = len(close)
    signal = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if close[i] <= 0 or ema_major[i] <= 0:
            signal[i] = 0.0
            continue
        
        # Primary trend direction from EMA crossover
        ema_diff_pct = (ema_fast[i] - ema_slow[i]) / close[i]
        ema_direction = np.sign(ema_diff_pct)
        
        # Major trend filter (price vs 100 EMA)
        major_filter = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend
        if ema_direction != major_filter and abs(ema_direction) > 0:
            # Conflicting signals → no trade
            signal[i] = 0.0
            continue
        
        # Calculate trend strength
        trend_strength = min(1.0, abs(ema_diff_pct) * 100)
        
        # RSI confirmation
        rsi_ok = False
        if ema_direction > 0:
            # Long: RSI must be in bullish but not overbought zone
            if RSI_LONG_MIN <= rsi[i] <= RSI_LONG_MAX:
                rsi_ok = True
        elif ema_direction < 0:
            # Short: RSI must be in bearish but not oversold zone
            if RSI_SHORT_MIN <= rsi[i] <= RSI_SHORT_MAX:
                rsi_ok = True
        
        if not rsi_ok:
            signal[i] = 0.0
            continue
        
        # RSI momentum factor (stronger when RSI confirms trend)
        if ema_direction > 0:
            rsi_factor = (rsi[i] - 50) / 20  # 0.0 to 1.0 in valid range
        else:
            rsi_factor = (50 - rsi[i]) / 20  # 0.0 to 1.0 in valid range
        
        rsi_factor = np.clip(rsi_factor, 0.3, 1.0)
        
        signal[i] = ema_direction * trend_strength * rsi_factor
    
    return np.clip(signal, -1.0, 1.0)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Volatility Regime Trend V13 Strategy.
    
    Signal Logic:
    1. Calculate trend signal from EMA crossover (12/26) with 100 EMA filter
    2. Check volatility regime (only trade in middle 50% of ATR range)
    3. Apply RSI momentum confirmation
    4. Reduce signal strength when funding is extreme
    5. Apply volume filter
    6. Smooth signals with EMA and hysteresis
    7. Filter by minimum signal magnitude
    
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
    
    # Calculate ATR as percentage of price
    atr_pct = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if close[i] > 0:
            atr_pct[i] = atr[i] / close[i]
        else:
            atr_pct[i] = 0.0
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    volatility_regime = calculate_volatility_regime(
        atr_pct, ATR_LOOKBACK, 
        VOLATILITY_LOWER_PERCENTILE, VOLATILITY_UPPER_PERCENTILE
    )
    funding_impact = calculate_funding_impact(
        funding_rate, FUNDING_EXTREME_THRESHOLD, FUNDING_IMPACT
    )
    
    # Calculate trend signal (vectorized)
    trend_signal = calculate_trend_signal(
        close, ema_fast, ema_slow, ema_major, rsi
    )
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        ATR_LOOKBACK,
        VOLUME_LOOKBACK
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
        
        # Volatility regime filter (CRITICAL for drawdown control)
        if volatility_regime[i] < 1.0:
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
        
        # Get base trend signal
        raw_signal = trend_signal[i]
        
        # Apply funding impact (reduce strength when funding is extreme)
        raw_signal *= funding_impact[i]
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
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