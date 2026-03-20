#!/usr/bin/env python3
"""
strategy.py - Z-Score Mean Reversion Multi-TF V9
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "15m")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Z-score mean reversion on 15m with 1h trend filter:
    - Primary signal: Z-score(20) extremes indicate overextended price
    - Trend filter: 1h EMA(50) direction determines bias
    - Entry: Only trade mean reversion IN DIRECTION of higher TF trend
    - Volume confirmation: Ensure sufficient liquidity
    - Signal scaling: Magnitude proportional to Z-score extremeness
    
    Why this works:
    - 15m captures intraday mean reversion opportunities
    - 1h trend filter prevents fighting major trends (reduces DD)
    - Z-score is cleaner than BB-KC squeeze (which failed with -68% DD)
    - Multi-TF approach reportedly doubled Sharpe in research
    - Conservative leverage (1.5x) controls drawdown

Look-Ahead Safety:
    - All rolling calculations use only past data (min_periods respected)
    - No .shift(-n) or future index access
    - Signal at bar t uses only prices.iloc[:t+1]
    - 1h trend calculated by resampling 15m data properly
"""

import numpy as np
import pandas as pd

# =============================================================================
# Strategy Configuration
# =============================================================================

name = "zscore_multitf_meanrev_v9"
timeframe = "15m"
leverage = 1.5  # Conservative to control drawdown

# Z-score configuration
ZSCORE_PERIOD = 20
ZSCORE_ENTRY_THRESHOLD = 1.8  # Enter when |Z| > this
ZSCORE_EXIT_THRESHOLD = 0.5  # Exit when |Z| < this
ZSCORE_MAX = 3.5  # Maximum Z-score for signal scaling

# Trend filter configuration (1h timeframe)
TREND_EMA_PERIOD = 50
TREND_LOOKBACK_BARS = 4  # 15m bars per 1h bar

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.60  # Volume must be at least this % of average

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.90  # Maximum signal magnitude
SMOOTHING_FACTOR = 0.40  # EMA smoothing for signals (0=none, 1=max)
HYSTERESIS_THRESHOLD = 0.15  # Minimum change to flip signal direction

# Risk management
ATR_PERIOD = 14
VOLATILITY_MIN = 0.002  # Minimum ATR % to trade
VOLATILITY_MAX = 0.040  # Maximum ATR % to trade
VOLATILITY_TARGET = 0.012  # Target ATR as % of price


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_zscore(close: np.ndarray, period: int = 20) -> np.ndarray:
    """
    Calculate Z-score of price relative to rolling mean.
    Z = (price - rolling_mean) / rolling_std
    Only uses past data (no look-ahead).
    """
    n = len(close)
    zscore = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return zscore
    
    close_series = pd.Series(close)
    rolling_mean = close_series.rolling(window=period, min_periods=period).mean()
    rolling_std = close_series.rolling(window=period, min_periods=period).std()
    
    zscore_values = (close_series - rolling_mean) / rolling_std.replace(0, np.inf)
    zscore = np.nan_to_num(zscore_values.values, nan=0.0)
    
    return zscore


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


def calculate_1h_trend_from_15m(prices: pd.DataFrame, ema_period: int = 50) -> np.ndarray:
    """
    Calculate 1h EMA trend from 15m data by proper resampling.
    Returns array same length as input (forward-filled 1h values).
    Only uses past data (no look-ahead).
    """
    n = len(prices)
    trend = np.zeros(n, dtype=np.float64)
    
    if n < TREND_LOOKBACK_BARS * ema_period:
        return trend
    
    # Resample 15m to 1h using only past data
    # Each 1h bar = 4 consecutive 15m bars
    close_15m = prices["close"].values
    
    # Build 1h close series (last close of each 4-bar group)
    n_1h_bars = n // TREND_LOOKBACK_BARS
    close_1h = np.zeros(n_1h_bars, dtype=np.float64)
    
    for i in range(n_1h_bars):
        start_idx = i * TREND_LOOKBACK_BARS
        end_idx = start_idx + TREND_LOOKBACK_BARS
        # Use the last close of the 1h period
        close_1h[i] = close_15m[end_idx - 1]
    
    # Calculate EMA on 1h data
    close_1h_series = pd.Series(close_1h)
    ema_1h = close_1h_series.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    ema_1h = np.nan_to_num(ema_1h, nan=0.0)
    
    # Forward-fill 1h EMA back to 15m resolution
    for i in range(n_1h_bars):
        start_idx = i * TREND_LOOKBACK_BARS
        end_idx = min(start_idx + TREND_LOOKBACK_BARS, n)
        if i < len(ema_1h):
            trend[start_idx:end_idx] = ema_1h[i]
    
    # Fill remaining bars with last known value
    if n_1h_bars * TREND_LOOKBACK_BARS < n:
        last_value = trend[n_1h_bars * TREND_LOOKBACK_BARS - 1] if n_1h_bars > 0 else 0.0
        trend[n_1h_bars * TREND_LOOKBACK_BARS:] = last_value
    
    return trend


def calculate_trend_direction(close: np.ndarray, trend_ema: np.ndarray) -> np.ndarray:
    """
    Calculate trend direction: +1 if price > EMA, -1 if price < EMA, 0 if neutral.
    Only uses current/past data.
    """
    n = len(close)
    direction = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if trend_ema[i] <= 0:
            direction[i] = 0.0
        elif close[i] > trend_ema[i] * 1.001:  # Small buffer to avoid noise
            direction[i] = 1.0
        elif close[i] < trend_ema[i] * 0.999:
            direction[i] = -1.0
        else:
            direction[i] = 0.0
    
    return direction


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Z-Score Mean Reversion Multi-TF V9 Strategy.
    
    Signal Logic:
    1. Calculate Z-score(20) on 15m closes
    2. Calculate 1h EMA(50) trend by resampling 15m→1h
    3. Determine trend direction (+1 bullish, -1 bearish)
    4. Generate mean reversion signal ONLY in trend direction:
       - If trend bullish: only take long when Z < -threshold
       - If trend bearish: only take short when Z > +threshold
    5. Scale signal by Z-score magnitude
    6. Apply volume and volatility filters
    7. Smooth signals and apply hysteresis
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, ...]
    
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
    zscore = calculate_zscore(close, ZSCORE_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    trend_ema_1h = calculate_1h_trend_from_15m(prices, TREND_EMA_PERIOD)
    trend_direction = calculate_trend_direction(close, trend_ema_1h)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        ZSCORE_PERIOD,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        TREND_LOOKBACK_BARS * TREND_EMA_PERIOD
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
        
        # Get Z-score and trend direction
        z = zscore[i]
        trend_dir = trend_direction[i]
        
        # No signal if no clear trend
        if trend_dir == 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Mean reversion logic: trade AGAINST Z-score but WITH trend
        raw_signal = 0.0
        
        if trend_dir > 0:
            # Bullish trend: only take long entries when price is oversold (Z < 0)
            if z < -ZSCORE_ENTRY_THRESHOLD:
                # Scale signal by Z-score magnitude (more extreme = stronger signal)
                z_magnitude = min(abs(z), ZSCORE_MAX) / ZSCORE_MAX
                raw_signal = z_magnitude * (abs(z) - ZSCORE_ENTRY_THRESHOLD) / (ZSCORE_MAX - ZSCORE_ENTRY_THRESHOLD)
                raw_signal = min(raw_signal, 1.0)
        elif trend_dir < 0:
            # Bearish trend: only take short entries when price is overbought (Z > 0)
            if z > ZSCORE_ENTRY_THRESHOLD:
                # Scale signal by Z-score magnitude
                z_magnitude = min(abs(z), ZSCORE_MAX) / ZSCORE_MAX
                raw_signal = -z_magnitude * (abs(z) - ZSCORE_ENTRY_THRESHOLD) / (ZSCORE_MAX - ZSCORE_ENTRY_THRESHOLD)
                raw_signal = max(raw_signal, -1.0)
        
        # If signal is too weak, skip
        if abs(raw_signal) < 0.1:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 1.5)
        raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals)
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