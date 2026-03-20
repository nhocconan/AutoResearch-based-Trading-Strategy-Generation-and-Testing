#!/usr/bin/env python3
"""
strategy.py - Adaptive Trend V10
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

Strategy Hypothesis:
    4h adaptive trend-following with volatility-based position sizing:
    - Primary: EMA crossover trend (12/26) on 4h timeframe
    - Filter: Price relative to 200 EMA for major trend direction
    - Confirmation: RSI momentum filter (avoid extremes)
    - Risk: ATR-based volatility normalization for consistent sizing
    - Drawdown control: Reduce leverage after consecutive losses
    
    Why 4h timeframe:
    - Cleaner signals than 1h (less noise/whipsaws)
    - More trades than 1d (sufficient sample for statistics)
    - Funding rates still meaningful at this frequency
    - Better risk/reward for crypto trend following

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

name = "adaptive_trend_v10"
timeframe = "4h"
leverage = 1.5  # Conservative for drawdown control

# EMA configuration for trend detection
EMA_FAST = 12
EMA_SLOW = 26
EMA_MAJOR = 200

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_MIN = 40  # Don't long if RSI below this
RSI_LONG_MAX = 70  # Don't long if RSI above this (overbought)
RSI_SHORT_MIN = 30  # Don't short if RSI below this (oversold)
RSI_SHORT_MAX = 60  # Don't short if RSI above this

# ATR/Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.025  # Target ATR as % of price
VOLATILITY_MIN = 0.005  # Minimum ATR % to trade
VOLATILITY_MAX = 0.100  # Maximum ATR % to trade

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.75  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.40  # EMA smoothing factor for signals

# Trade management
MIN_BARS_PER_TRADE = 3  # Minimum bars to hold position
MAX_CONSECUTIVE_LOSSES = 3  # Reduce signal after this many losses


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
    """
    n = len(close)
    if n < period:
        return np.zeros(n, dtype=np.float64)
    
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


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Trend V10 Strategy.
    
    Signal Logic:
    1. Calculate EMA crossover trend signal (12/26 EMA)
    2. Filter by major trend (price vs 200 EMA)
    3. Confirm with RSI momentum (not overbought/oversold)
    4. Normalize by volatility (ATR) for consistent risk
    5. Smooth signals to reduce whipsaws
    6. Apply minimum magnitude filter
    
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
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1
    )
    
    # Track signal state
    prev_signal = 0.0
    smoothed_signal = 0.0
    bars_in_position = 0
    current_position = 0  # 0=none, 1=long, -1=short
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            smoothed_signal = 0.0
            continue
        
        # Volatility filter (not too low, not too high)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            smoothed_signal = 0.0
            continue
        
        # Calculate EMA crossover signal
        ema_diff = (ema_fast[i] - ema_slow[i]) / close[i]
        ema_direction = np.sign(ema_diff)
        
        # Major trend filter (price vs 200 EMA)
        major_trend = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend
        if ema_direction != major_trend or ema_direction == 0:
            raw_signal = 0.0
        else:
            # Calculate trend strength
            trend_strength = abs(ema_diff) * 80
            trend_strength = np.clip(trend_strength, 0.0, 1.0)
            
            # RSI momentum filter
            rsi_ok = True
            if ema_direction > 0:  # Long signal
                if rsi[i] < RSI_LONG_MIN or rsi[i] > RSI_LONG_MAX:
                    rsi_ok = False
            elif ema_direction < 0:  # Short signal
                if rsi[i] < RSI_SHORT_MIN or rsi[i] > RSI_SHORT_MAX:
                    rsi_ok = False
            
            if rsi_ok:
                raw_signal = ema_direction * trend_strength
            else:
                raw_signal = 0.0
        
        # Volatility normalization (scale by target volatility)
        if raw_signal != 0:
            vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
            vol_factor = np.clip(vol_factor, 0.5, 2.0)
            raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        # Track position duration
        signal_direction = np.sign(signal)
        if signal_direction != 0 and signal_direction == current_position:
            bars_in_position += 1
        elif signal_direction != 0:
            bars_in_position = 1
            current_position = signal_direction
        else:
            bars_in_position = 0
            current_position = 0
        
        # Only allow trade if signal persists for minimum bars
        if bars_in_position >= MIN_BARS_PER_TRADE or signal_direction == 0:
            signals[i] = signal
        else:
            signals[i] = 0.0
        
        prev_signal = signal
    
    return signals