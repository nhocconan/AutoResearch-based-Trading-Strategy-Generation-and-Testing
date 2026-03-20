#!/usr/bin/env python3
"""
strategy.py - HMA Trend Momentum V16
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "4h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Hull Moving Average trend-following on 4h timeframe:
    - HMA(16) vs HMA(48) crossover for trend direction
    - HMA slope confirmation (rising/falling)
    - ATR volatility filter to avoid low-vol chop
    - Price vs HMA(48) position for trend strength
    - Simple momentum confirmation via ROC(10)
    
    Why this should work better than previous attempts:
    - HMA has less lag than EMA while maintaining smoothness
    - 4h timeframe captures sustained trends without noise
    - Fewer filters than v12-v15 = more actual trades
    - No funding rate dependency (works across all symbols)
    - ATR filter avoids whipsaws in low volatility
    - Conservative leverage (1.5x) for better risk-adjusted returns
    
    Key improvements over failed strategies:
    - Simpler signal logic (avoided over-filtering like v12-v15)
    - No mean-reversion component (crypto trends persist)
    - Proper HMA calculation (unlike failed hma_regime_hybrid_v1)
    - Volatility-adjusted position sizing

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

name = "hma_trend_momentum_v16"
timeframe = "4h"
leverage = 1.5  # Conservative for better Sharpe ratio

# HMA configuration
HMA_FAST = 16
HMA_SLOW = 48
HMA_TREND = 48  # For trend filter

# Momentum configuration
ROC_PERIOD = 10
ROC_THRESHOLD = 0.02  # 2% momentum threshold

# Volatility configuration
ATR_PERIOD = 14
ATR_MIN_PCT = 0.005  # Minimum ATR % to trade (avoid low vol)
ATR_MAX_PCT = 0.080  # Maximum ATR % to trade (avoid extreme vol)
VOLATILITY_TARGET = 0.020  # Target volatility for position sizing

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.90  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.30  # EMA smoothing factor for signals
TRENGTH_THRESHOLD = 0.15  # Minimum trend strength

# Risk management
MAX_DRAWDOWN_TARGET = 0.40  # Target max drawdown


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_wma(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Weighted Moving Average using only past data.
    """
    n = len(close)
    wma = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return wma
    
    weights = np.arange(1, period + 1, dtype=np.float64)
    weights = weights / weights.sum()
    
    close_series = pd.Series(close)
    wma_series = close_series.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights), raw=True
    )
    
    wma = np.nan_to_num(wma_series.values, nan=0.0)
    
    return wma


def calculate_hma(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Hull Moving Average using only past data.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    hma = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return hma
    
    half_period = period // 2
    if half_period < 1:
        half_period = 1
    
    sqrt_period = int(np.sqrt(period))
    if sqrt_period < 1:
        sqrt_period = 1
    
    # Calculate WMA(n/2) and WMA(n)
    wma_half = calculate_wma(close, half_period)
    wma_full = calculate_wma(close, period)
    
    # Calculate 2*WMA(n/2) - WMA(n)
    diff = 2.0 * wma_half - wma_full
    
    # Calculate WMA of the difference with sqrt(n) period
    hma = calculate_wma(diff, sqrt_period)
    
    return hma


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


def calculate_roc(close: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Calculate Rate of Change using only past data.
    ROC = (close - close[period]) / close[period]
    """
    n = len(close)
    roc = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return roc
    
    close_series = pd.Series(close)
    roc_series = close_series.pct_change(periods=period)
    
    roc = np.nan_to_num(roc_series.values, nan=0.0)
    
    return roc


def calculate_hma_slope(hma: np.ndarray, lookback: int = 3) -> np.ndarray:
    """
    Calculate HMA slope (rate of change over lookback periods).
    Positive = rising, Negative = falling.
    Only uses past HMA values (no look-ahead).
    """
    n = len(hma)
    slope = np.zeros(n, dtype=np.float64)
    
    if n < lookback + 1:
        return slope
    
    for i in range(lookback, n):
        if hma[i-lookback] != 0:
            slope[i] = (hma[i] - hma[i-lookback]) / hma[i-lookback]
        else:
            slope[i] = 0.0
    
    return slope


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    HMA Trend Momentum V16 Strategy.
    
    Signal Logic:
    1. Calculate HMA(16) and HMA(48) for trend direction
    2. Calculate HMA slope for momentum confirmation
    3. Calculate ROC(10) for additional momentum
    4. Calculate ATR for volatility filter
    5. Combine signals with proper weighting
    6. Apply volatility normalization
    7. Smooth signals to reduce whipsaws
    
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
    hma_fast = calculate_hma(close, HMA_FAST)
    hma_slow = calculate_hma(close, HMA_SLOW)
    
    hma_slope_fast = calculate_hma_slope(hma_fast, lookback=3)
    hma_slope_slow = calculate_hma_slope(hma_slow, lookback=5)
    
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    roc = calculate_roc(close, ROC_PERIOD)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        HMA_SLOW * 2,  # HMA needs extra warmup due to nested WMA
        ATR_PERIOD + 1,
        ROC_PERIOD + 1,
        10  # Minimum for slope calculation
    )
    
    # Generate signals
    prev_signal = 0.0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volatility filter (not too low, not too high)
        atr_pct = atr[i] / close[i]
        if atr_pct < ATR_MIN_PCT or atr_pct > ATR_MAX_PCT:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate HMA crossover signal
        hma_diff = (hma_fast[i] - hma_slow[i]) / close[i]
        hma_direction = np.sign(hma_diff)
        hma_strength = abs(hma_diff) * 100  # Scale to reasonable range
        
        # HMA slope confirmation
        slope_confirmation = 1.0
        if hma_direction > 0:
            # Long: want rising HMA
            if hma_slope_fast[i] <= 0 or hma_slope_slow[i] <= 0:
                slope_confirmation = 0.5  # Reduce strength if slope not confirmed
        elif hma_direction < 0:
            # Short: want falling HMA
            if hma_slope_fast[i] >= 0 or hma_slope_slow[i] >= 0:
                slope_confirmation = 0.5  # Reduce strength if slope not confirmed
        
        # Price position vs HMA(48)
        price_vs_hma = (close[i] - hma_slow[i]) / close[i]
        price_position = np.sign(price_vs_hma)
        
        # Momentum confirmation via ROC
        momentum_confirmation = 1.0
        if hma_direction > 0:
            # Long: want positive ROC
            if roc[i] < 0:
                momentum_confirmation = 0.3  # Weak momentum
            elif roc[i] < ROC_THRESHOLD:
                momentum_confirmation = 0.7  # Moderate momentum
        elif hma_direction < 0:
            # Short: want negative ROC
            if roc[i] > 0:
                momentum_confirmation = 0.3  # Weak momentum
            elif roc[i] > -ROC_THRESHOLD:
                momentum_confirmation = 0.7  # Moderate momentum
        
        # Check trend alignment (all signals should agree)
        trend_aligned = (
            np.sign(hma_direction) == np.sign(price_position) or
            abs(price_vs_hma) < 0.01  # Price near HMA is okay
        )
        
        if not trend_aligned:
            # Conflicting signals → reduce strength significantly
            trend_strength = hma_strength * 0.3
        else:
            # Aligned signals → full strength
            trend_strength = hma_strength
        
        # Combine all factors
        raw_signal = (
            hma_direction * 
            trend_strength * 
            slope_confirmation * 
            momentum_confirmation
        )
        
        # Apply minimum strength threshold
        if abs(raw_signal) < TRENGTH_THRESHOLD:
            raw_signal = 0.0
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals to reduce whipsaws)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals