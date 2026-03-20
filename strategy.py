#!/usr/bin/env python3
"""
strategy.py - Volatility Trend V1
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "4h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simplified trend-following with volatility-based risk control:
    - Primary signal: EMA crossover (12/26) for trend direction
    - Filter: Price above/below 100 EMA for major trend validation
    - Risk control: ATR-based position sizing (reduce exposure in high vol)
    - Funding overlay: Only at extreme levels (contrarian signal)
    - Smoothing: Signal EMA to reduce whipsaws
    
    Why 4h timeframe:
    - Cleaner trends than 1h (less noise)
    - More trades than 1d (better statistics)
    - Lower transaction cost impact than 5m/15m
    - Works well across BTC/ETH/SOL volatility profiles

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

name = "volatility_trend_v1"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for drawdown control

# EMA configuration for trend detection
EMA_FAST = 12
EMA_SLOW = 26
EMA_MAJOR = 100

# ATR configuration for volatility control
ATR_PERIOD = 14
ATR_VOLATILITY_TARGET = 0.02  # Target ATR as % of price
ATR_VOLATILITY_MIN = 0.005  # Minimum ATR % to trade
ATR_VOLATILITY_MAX = 0.06  # Maximum ATR % to trade

# Funding rate configuration (simplified)
FUNDING_EXTREME_THRESHOLD = 0.0015  # 0.15% per 8hr = very extreme
FUNDING_WEIGHT = 0.25  # Reduced weight - trend is primary

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.40  # EMA smoothing factor for signals
DIRECTION_CHANGE_MIN = 0.15  # Minimum change to flip direction


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


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using only past data.
    """
    n = len(close)
    if n < period + 1:
        return np.zeros(n, dtype=np.float64)
    
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


def calculate_funding_signal(funding_rate: np.ndarray, 
                             threshold: float = 0.0015,
                             weight: float = 0.25) -> np.ndarray:
    """
    Calculate funding rate contrarian signal.
    Only activates at extreme funding levels.
    Returns value in [-weight, weight].
    """
    n = len(funding_rate)
    signal = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        fr = funding_rate[i]
        
        if fr > threshold:
            # Extreme positive funding → short bias
            signal[i] = -weight * min(1.0, fr / threshold)
        elif fr < -threshold:
            # Extreme negative funding → long bias
            signal[i] = weight * min(1.0, abs(fr) / threshold)
        else:
            signal[i] = 0.0
    
    return signal


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Volatility Trend V1 Strategy.
    
    Signal Logic:
    1. Calculate EMA crossover signal (12/26)
    2. Filter by major trend (100 EMA)
    3. Scale by volatility (ATR)
    4. Add funding overlay at extremes only
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
    
    # Fix invalid prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators (all use only past data)
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    funding_signal = calculate_funding_signal(funding_rate, FUNDING_EXTREME_THRESHOLD, FUNDING_WEIGHT)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(EMA_MAJOR, EMA_SLOW, ATR_PERIOD + 1)
    
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
        if atr_pct < ATR_VOLATILITY_MIN or atr_pct > ATR_VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Calculate trend direction from EMA crossover
        ema_diff = ema_fast[i] - ema_slow[i]
        ema_direction = np.sign(ema_diff)
        
        # Major trend filter (price vs 100 EMA)
        major_filter = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend
        if ema_direction != major_filter and abs(ema_direction) > 0:
            # Conflicting signals → reduce strength significantly
            trend_strength = abs(ema_diff) / close[i] * 30 * 0.3
        else:
            # Aligned signals → full strength
            trend_strength = abs(ema_diff) / close[i] * 30
        
        # Base signal from trend
        raw_signal = ema_direction * trend_strength
        
        # Add funding overlay (only at extremes)
        fund_sig = funding_signal[i]
        if abs(fund_sig) > 0.05:
            # Funding at extreme - apply contrarian overlay
            if np.sign(raw_signal) != np.sign(fund_sig):
                # Conflict - reduce trend signal
                raw_signal = raw_signal * 0.7 + fund_sig
            else:
                # Aligned - slight reinforcement
                raw_signal = raw_signal * 0.85 + fund_sig * 0.15
        
        # Volatility normalization (reduce position size in high vol)
        vol_factor = ATR_VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.4, 1.8)
        raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # Hysteresis: don't flip direction on small changes
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < DIRECTION_CHANGE_MIN:
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