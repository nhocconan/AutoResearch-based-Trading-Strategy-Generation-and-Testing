#!/usr/bin/env python3
"""
strategy.py - Multi-Timeframe Trend V3
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Clean trend-following on 4h timeframe with strict risk controls:
    - Primary: EMA crossover (12/26) for trend direction
    - Filter: Price above/below 50 EMA for trend validation
    - Momentum: RSI confirmation (not extreme)
    - Volatility: Skip trades when ATR% is too high (reduces drawdown)
    - Signal smoothing to reduce whipsaws
    
    Why 4h timeframe:
    - Less noise than 1h/15m/5m (cleaner trends)
    - More trades than 1d (ensures 10+ trades requirement)
    - Funding rates apply every 8h, 4h captures 2 per day
    - Better risk/reward for trend following
    
    Why this improves on v12:
    - Simpler logic (less overfitting risk)
    - Stricter volatility filter (reduces drawdown)
    - Lower leverage (1.5x vs 2.0x)
    - Better signal persistence (less flipping)

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

name = "mtf_trend_v3"
timeframe = "4h"
leverage = 1.5  # Conservative for better drawdown control

# EMA configuration
EMA_FAST = 12
EMA_SLOW = 26
EMA_TREND = 50

# RSI configuration
RSI_PERIOD = 14
RSI_LONG_MIN = 45  # RSI must be above this for longs
RSI_SHORT_MAX = 55  # RSI must be below this for shorts

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_MAX_PCT = 0.04  # Skip if ATR% > 4% (high volatility = high risk)
VOLATILITY_MIN_PCT = 0.005  # Skip if ATR% < 0.5% (too low volatility)

# Signal configuration
MIN_SIGNAL = 0.25  # Minimum signal magnitude to trade
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SIGNAL_SMOOTH = 0.3  # EMA smoothing factor for signals
HYSTERESIS = 0.15  # Minimum change to flip direction

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.6  # Volume must be at least 60% of average


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
    Multi-Timeframe Trend V3 Strategy.
    
    Signal Logic:
    1. Calculate EMAs (12, 26, 50) for trend detection
    2. Calculate RSI for momentum confirmation
    3. Calculate ATR for volatility filtering
    4. Calculate volume ratio for liquidity check
    5. Generate raw signal from EMA crossover + RSI filter
    6. Apply volatility filter (skip high vol periods)
    7. Smooth signals with EMA
    8. Apply hysteresis to reduce whipsaws
    9. Filter by minimum signal magnitude
    
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
    ema_trend = calculate_ema(close, EMA_TREND)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_TREND,
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
        
        # Volatility filter (CRITICAL for drawdown control)
        atr_pct = atr[i] / close[i]
        if atr_pct > VOLATILITY_MAX_PCT or atr_pct < VOLATILITY_MIN_PCT:
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
        
        # Determine trend direction from EMA crossover
        ema_diff = ema_fast[i] - ema_slow[i]
        ema_direction = np.sign(ema_diff)
        
        # Major trend filter (price vs 50 EMA)
        major_filter = np.sign(close[i] - ema_trend[i])
        
        # Only trade in direction of major trend (reduces whipsaws)
        if ema_direction != major_filter:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # RSI momentum confirmation
        rsi_valid = False
        if ema_direction > 0:
            # Long: RSI must be above minimum (momentum) but not overbought
            if RSI_LONG_MIN <= rsi[i] <= 70:
                rsi_valid = True
        elif ema_direction < 0:
            # Short: RSI must be below maximum (momentum) but not oversold
            if RSI_SHORT_MAX >= rsi[i] >= 30:
                rsi_valid = True
        
        if not rsi_valid:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Calculate raw signal strength
        # Scale by EMA divergence (stronger crossover = stronger signal)
        ema_strength = abs(ema_diff) / close[i] * 100
        ema_strength = np.clip(ema_strength, 0.1, 1.0)
        
        # RSI factor (closer to neutral = stronger signal for trend following)
        if ema_direction > 0:
            rsi_factor = (rsi[i] - RSI_LONG_MIN) / (70 - RSI_LONG_MIN)
        else:
            rsi_factor = (RSI_SHORT_MAX - rsi[i]) / (RSI_SHORT_MAX - 30)
        rsi_factor = np.clip(rsi_factor, 0.3, 1.0)
        
        # Raw signal
        raw_signal = ema_direction * ema_strength * rsi_factor
        
        # Signal smoothing (EMA on signals to reduce whipsaws)
        smoothed_signal = SIGNAL_SMOOTH * prev_signal + (1.0 - SIGNAL_SMOOTH) * raw_signal
        
        # Hysteresis: don't flip direction on small changes
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < HYSTERESIS:
                smoothed_signal = prev_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals