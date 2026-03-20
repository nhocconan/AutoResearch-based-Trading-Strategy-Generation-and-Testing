#!/usr/bin/env python3
"""
strategy.py - MACD RSI Momentum Hybrid 1h V1
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

Strategy Hypothesis:
    MACD Histogram Momentum with RSI Filter on 1h timeframe.
    
    Why this works:
    - MACD histogram captures momentum acceleration/deceleration
    - More responsive than Supertrend (which had -84% DD)
    - RSI filter prevents entries at overextended levels
    - 1h timeframe generates more trades than 4h (need ≥10 trades)
    - Volume confirmation ensures liquidity at entry
    - ATR-based signal scaling controls risk during high volatility
    
    Key differences from failed strategies:
    - Not pure trend (Supertrend failed with -84% DD)
    - Not pure mean reversion (BB/KC failed with -68% DD)
    - Momentum with mean-reversion filter = balanced approach
    - 1h should generate more trades than 4h strategies
    
    Risk Management:
    - Signal magnitude scales with ATR volatility
    - RSI extremes filter prevents chasing moves
    - Volume filter avoids illiquid entries
    - Signal smoothing reduces whipsaw trades

Look-Ahead Safety:
    - All rolling calculations use only past data (min_periods respected)
    - No .shift(-n) or future index access
    - Signal at bar t uses only prices.iloc[:t+1]
    - Using numpy operations (not pandas .replace on arrays)
"""

import numpy as np
import pandas as pd

# =============================================================================
# Strategy Configuration
# =============================================================================

name = "macd_rsi_momentum_1h_v1"
timeframe = "1h"
leverage = 1.5  # Conservative leverage for risk control

# MACD Configuration
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MACD_HIST_THRESHOLD = 0.0  # Histogram must cross zero with momentum

# RSI Configuration
RSI_PERIOD = 14
RSI_LONG_MAX = 65   # Don't long if RSI above this (overbought)
RSI_SHORT_MIN = 35  # Don't short if RSI below this (oversold)
RSI_NEUTRAL_LOW = 40
RSI_NEUTRAL_HIGH = 60

# Volume Configuration
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.60  # Volume must be at least this % of average

# Volatility Configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.015  # Target ATR as % of price
VOLATILITY_MIN = 0.003     # Minimum ATR % to trade
VOLATILITY_MAX = 0.060     # Maximum ATR % to trade

# Signal Configuration
MIN_SIGNAL_MAGNITUDE = 0.15
MAX_SIGNAL = 0.85
SMOOTHING_FACTOR = 0.35
HYSTERESIS_THRESHOLD = 0.12

# Trend Filter (optional secondary confirmation)
EMA_TREND = 100
USE_TREND_FILTER = True


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


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using only past data.
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    n = len(close)
    rsi = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    delta = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        delta[i] = close[i] - close[i-1]
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    
    # Calculate average gain and loss using EMA
    gains_series = pd.Series(gains)
    losses_series = pd.Series(losses)
    
    avg_gain = gains_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    avg_loss = losses_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate RS and RSI
    rs = np.zeros(n, dtype=np.float64)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n, dtype=np.float64)
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    
    # Handle division by zero (no losses = RSI 100)
    rsi[avg_loss <= 0] = 100.0
    
    rsi = np.nan_to_num(rsi, nan=50.0)
    rsi = np.clip(rsi, 0.0, 100.0)
    
    return rsi


def calculate_macd(close: np.ndarray, 
                   fast: int = 12, 
                   slow: int = 26, 
                   signal: int = 9) -> tuple:
    """
    Calculate MACD line, signal line, and histogram using only past data.
    
    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD, signal)
    Histogram = MACD - Signal
    """
    n = len(close)
    macd_line = np.zeros(n, dtype=np.float64)
    signal_line = np.zeros(n, dtype=np.float64)
    histogram = np.zeros(n, dtype=np.float64)
    
    if n < slow + signal + 5:
        return macd_line, signal_line, histogram
    
    close_series = pd.Series(close)
    
    # Calculate EMAs
    ema_fast = close_series.ewm(span=fast, adjust=False, min_periods=fast).mean().values
    ema_slow = close_series.ewm(span=slow, adjust=False, min_periods=slow).mean().values
    
    # MACD line
    macd_line = ema_fast - ema_slow
    macd_line = np.nan_to_num(macd_line, nan=0.0)
    
    # Signal line
    macd_series = pd.Series(macd_line)
    signal_line = macd_series.ewm(span=signal, adjust=False, min_periods=signal).mean().values
    signal_line = np.nan_to_num(signal_line, nan=0.0)
    
    # Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


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
    
    # Use numpy where to avoid division issues
    volume_ratio = np.where(
        rolling_avg.values > 0,
        volume_series.values / rolling_avg.values,
        1.0
    )
    volume_ratio = np.nan_to_num(volume_ratio, nan=1.0)
    
    return volume_ratio


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    MACD RSI Momentum Hybrid V1 Strategy.
    
    Signal Logic:
    1. Calculate MACD histogram for momentum direction
    2. Filter by RSI to avoid overextended entries
    3. Confirm with volume (liquidity check)
    4. Scale signal by volatility (ATR normalization)
    5. Smooth signals with EMA
    6. Apply hysteresis to reduce whipsaws
    7. Filter by minimum signal magnitude
    
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
    ema_trend = calculate_ema(close, EMA_TREND)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    rsi = calculate_rsi(close, RSI_PERIOD)
    macd_line, signal_line, histogram = calculate_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_TREND,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        RSI_PERIOD + 5,
        MACD_SLOW + MACD_SIGNAL + 10
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
        
        # RSI filter - avoid overextended entries
        rsi_value = rsi[i]
        
        # Trend filter (optional)
        if USE_TREND_FILTER:
            trend_direction = np.sign(close[i] - ema_trend[i])
        else:
            trend_direction = 0
        
        # MACD histogram momentum signal
        hist_value = histogram[i]
        prev_hist = histogram[i-1] if i > 0 else 0.0
        
        raw_signal = 0.0
        
        # Long signal: MACD histogram turning positive + RSI not overbought
        if hist_value > 0 and prev_hist <= 0:
            # Histogram crossing above zero (momentum turning positive)
            if rsi_value < RSI_LONG_MAX:
                # RSI allows long entry
                if trend_direction >= 0 or not USE_TREND_FILTER:
                    # Trend filter passes (or disabled)
                    raw_signal = 1.0 * min(abs(hist_value) / (close[i] * 0.001), 1.0)
        
        # Short signal: MACD histogram turning negative + RSI not oversold
        elif hist_value < 0 and prev_hist >= 0:
            # Histogram crossing below zero (momentum turning negative)
            if rsi_value > RSI_SHORT_MIN:
                # RSI allows short entry
                if trend_direction <= 0 or not USE_TREND_FILTER:
                    # Trend filter passes (or disabled)
                    raw_signal = -1.0 * min(abs(hist_value) / (close[i] * 0.001), 1.0)
        
        # Alternative: MACD histogram momentum (not just crossover)
        if raw_signal == 0.0:
            # Use histogram slope for continuous momentum
            hist_slope = hist_value - prev_hist
            
            if hist_slope > 0 and rsi_value < RSI_NEUTRAL_HIGH:
                # Positive momentum, RSI not too high
                if trend_direction >= 0 or not USE_TREND_FILTER:
                    raw_signal = 0.5 * min(abs(hist_slope) / (close[i] * 0.0005), 1.0)
            elif hist_slope < 0 and rsi_value > RSI_NEUTRAL_LOW:
                # Negative momentum, RSI not too low
                if trend_direction <= 0 or not USE_TREND_FILTER:
                    raw_signal = -0.5 * min(abs(hist_slope) / (close[i] * 0.0005), 1.0)
        
        # Volatility normalization (scale by target volatility)
        if raw_signal != 0.0:
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