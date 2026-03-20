#!/usr/bin/env python3
"""
strategy.py - Clean Trend 4h V1
====================================================================
Simple, robust trend-following strategy on 4h timeframe.

Strategy Hypothesis:
    - Dual EMA crossover (12/26) for trend direction
    - 200 EMA filter for major trend alignment
    - RSI filter to avoid entering at extremes
    - ATR-based volatility normalization
    - No funding rate dependency (more robust across symbols)
    
    Why 4h:
    - Cleaner signals than 1h/15m
    - More trades than 1d
    - Less noise, better risk/reward
    - Works well for BTC/ETH/SOL

Look-Ahead Safety:
    - All rolling calculations use only past data
    - No .shift(-n) or future index access
    - Signal at bar t uses only prices.iloc[:t+1]
"""

import numpy as np
import pandas as pd

# =============================================================================
# Strategy Configuration
# =============================================================================

name = "clean_trend_4h_v1"
timeframe = "4h"
leverage = 1.5  # Conservative for better Sharpe

# EMA configuration
EMA_FAST = 12
EMA_SLOW = 26
EMA_MAJOR = 200

# RSI configuration
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# ATR configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.02  # 2% target ATR

# Signal configuration
MIN_SIGNAL = 0.2  # Minimum signal magnitude to trade
MAX_SIGNAL = 0.8  # Maximum signal magnitude
SMOOTHING = 0.3  # Signal smoothing factor


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA using only past data."""
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return ema
    
    close_series = pd.Series(close)
    ema_values = close_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    ema = np.nan_to_num(ema_values, nan=0.0)
    
    return ema


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI using only past data."""
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
    """Calculate ATR using only past data."""
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
    Clean Trend 4h Strategy.
    
    Signal Logic:
    1. Calculate EMA crossover (12/26) for trend direction
    2. Filter by 200 EMA for major trend alignment
    3. Apply RSI filter to avoid extremes
    4. Normalize by ATR for volatility adjustment
    5. Smooth signals to reduce whipsaws
    
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
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Clean data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    
    # Fix invalid prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate indicators (all use only past data)
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(EMA_MAJOR, EMA_SLOW, RSI_PERIOD + 1, ATR_PERIOD + 1)
    
    # Generate signals
    prev_signal = 0.0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate ATR as percentage of price
        atr_pct = atr[i] / close[i]
        
        # Volatility filter (not too low, not too high)
        if atr_pct < 0.005 or atr_pct > 0.10:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate EMA crossover signal
        ema_diff = ema_fast[i] - ema_slow[i]
        ema_direction = np.sign(ema_diff)
        
        # Major trend filter (price vs 200 EMA)
        major_direction = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend
        if ema_direction != 0 and ema_direction != major_direction:
            # Conflicting signals → skip
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate trend strength (normalized by price)
        trend_strength = abs(ema_diff) / close[i] * 100
        trend_strength = np.clip(trend_strength, 0.0, 2.0)
        
        # RSI filter
        rsi_factor = 1.0
        if ema_direction > 0:
            # Long: avoid overbought
            if rsi[i] > RSI_OVERBOUGHT:
                rsi_factor = 0.3
            elif rsi[i] < 30:
                rsi_factor = 1.2  # Favorable for long
        elif ema_direction < 0:
            # Short: avoid oversold
            if rsi[i] < RSI_OVERSOLD:
                rsi_factor = 0.3
            elif rsi[i] > 70:
                rsi_factor = 1.2  # Favorable for short
        
        # Calculate raw signal
        raw_signal = ema_direction * trend_strength * rsi_factor
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        raw_signal *= vol_factor
        
        # Clip to [-1, 1]
        raw_signal = np.clip(raw_signal, -1.0, 1.0)
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SMOOTHING * prev_signal + (1.0 - SMOOTHING) * raw_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals