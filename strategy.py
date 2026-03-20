#!/usr/bin/env python3
"""
strategy.py - Trend Volatility Controlled V9
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    4h timeframe trend following with volatility-based position sizing:
    - Primary signal: EMA crossover (12/26) for trend direction
    - Major trend filter: Price above/below 200 EMA
    - Volatility adjustment: Scale signals by ATR to normalize risk
    - Signal smoothing: Reduce whipsaws with EMA on signals
    - Conservative leverage: 1.5x to control drawdown
    
    Why 4h timeframe:
    - Cleaner signals than 1h/15m (less noise)
    - More trades than 1d (sufficient sample size)
    - Better risk-adjusted returns in crypto trends
    - Reduced whipsaws that cause deep drawdowns

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

name = "trend_volatility_controlled_v9"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for drawdown control

# EMA configuration for trend detection
EMA_FAST = 12
EMA_SLOW = 26
EMA_MAJOR = 200

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_THRESHOLD = 40  # RSI must be above this for longs
RSI_SHORT_THRESHOLD = 60  # RSI must be below this for shorts

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.020  # Target ATR as % of price
VOLATILITY_MIN = 0.005  # Minimum ATR % to trade
VOLATILITY_MAX = 0.080  # Maximum ATR % to trade

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SMOOTHING_FACTOR = 0.40  # EMA smoothing for signals (0=none, 1=max)
HYSTERESIS_THRESHOLD = 0.15  # Minimum change to flip signal direction

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.50  # Volume must be at least this % of average


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


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Volatility Controlled V9 Strategy.
    
    Signal Logic:
    1. Calculate EMA crossover signal (12/26)
    2. Apply 200 EMA major trend filter
    3. Confirm with RSI momentum
    4. Scale by volatility (ATR)
    5. Smooth signals with EMA
    6. Apply hysteresis to reduce whipsaws
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
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
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
        
        # Calculate EMA crossover signal
        ema_diff = (ema_fast[i] - ema_slow[i]) / close[i]
        ema_direction = np.sign(ema_diff)
        
        # Major trend filter (price vs 200 EMA)
        major_filter = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend
        if ema_direction == 0 or major_filter == 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Check if trend aligns with major trend
        if ema_direction != major_filter:
            # Conflicting signals → skip trade
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # RSI momentum confirmation
        rsi_factor = 1.0
        if ema_direction > 0:
            # Long: want RSI above threshold but not overbought
            if rsi[i] < RSI_LONG_THRESHOLD:
                rsi_factor = 0.0  # No long entry
            elif rsi[i] > 75:
                rsi_factor = 0.5  # Reduce strength if overbought
        elif ema_direction < 0:
            # Short: want RSI below threshold but not oversold
            if rsi[i] > RSI_SHORT_THRESHOLD:
                rsi_factor = 0.0  # No short entry
            elif rsi[i] < 25:
                rsi_factor = 0.5  # Reduce strength if oversold
        
        if rsi_factor == 0.0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Calculate trend strength
        trend_strength = abs(ema_diff) * 100
        trend_strength = np.clip(trend_strength, 0.1, 1.0)
        
        # Base signal
        raw_signal = ema_direction * trend_strength * rsi_factor
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
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