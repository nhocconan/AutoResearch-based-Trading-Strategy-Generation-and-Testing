#!/usr/bin/env python3
"""
strategy.py - Adaptive Trend V3
====================================================================
Strategy Hypothesis:
    Clean trend-following on 4h timeframe with adaptive parameters:
    - Primary: EMA crossover (20/50) for trend direction
    - Filter: Price above/below 100 EMA for major trend confirmation
    - Entry: RSI momentum (not overbought/oversold extremes)
    - Volatility: ATR-based position sizing and filter
    - No funding rate dependency (more robust across all symbols)
    
    Why 4h timeframe:
    - Cleaner trends than 1h/15m (less noise)
    - Sufficient trade frequency (unlike 1d)
    - Better risk/reward for crypto trend following
    - Lower transaction cost impact vs lower timeframes

Look-Ahead Safety:
    - All indicators use only past data (min_periods respected)
    - No .shift(-n) or future index access
    - Signal at bar t uses only prices.iloc[:t+1]
"""

import numpy as np
import pandas as pd

# =============================================================================
# Strategy Configuration
# =============================================================================

name = "adaptive_trend_v3"
timeframe = "4h"
leverage = 1.5  # Conservative for better Sharpe

# EMA Configuration
EMA_FAST = 20
EMA_MED = 50
EMA_SLOW = 100

# RSI Configuration
RSI_PERIOD = 14
RSI_LONG_MIN = 40  # Minimum RSI for long entry
RSI_SHORT_MAX = 60  # Maximum RSI for short entry

# ATR Configuration
ATR_PERIOD = 14
ATR_MIN_PCT = 0.005  # Minimum volatility to trade
ATR_MAX_PCT = 0.080  # Maximum volatility (avoid extreme moves)

# Signal Configuration
MIN_SIGNAL = 0.25  # Minimum signal magnitude to trade
MAX_SIGNAL = 0.85  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.4  # EMA smoothing factor for signals
HYSTERESIS = 0.12  # Minimum change to flip direction

# Trend Strength
TREND_MIN_STRENGTH = 0.002  # Minimum EMA diff % to consider trend


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA using only past data."""
    n = len(close)
    if n < period:
        return np.zeros(n, dtype=np.float64)
    
    series = pd.Series(close)
    ema = series.ewm(span=period, adjust=False, min_periods=period).mean()
    return np.nan_to_num(ema.values, nan=0.0)


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI using only past data."""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0, dtype=np.float64)
    
    series = pd.Series(close)
    delta = series.diff()
    
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return np.nan_to_num(rsi.values, nan=50.0)


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                  period: int = 14) -> np.ndarray:
    """Calculate ATR using only past data."""
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
    atr = tr_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    return np.nan_to_num(atr.values, nan=0.0)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Trend V3 Strategy.
    
    Signal Logic:
    1. Calculate EMAs (20, 50, 100) for trend detection
    2. Determine trend direction from EMA relationship
    3. Confirm with major trend (price vs 100 EMA)
    4. Filter with RSI momentum
    5. Apply volatility filter (ATR)
    6. Smooth signals and apply hysteresis
    
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
    
    # Calculate all indicators (all use only past data)
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_med = calculate_ema(close, EMA_MED)
    ema_slow = calculate_ema(close, EMA_SLOW)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Minimum warmup period (all indicators need data)
    min_valid_index = max(EMA_SLOW, RSI_PERIOD + 1, ATR_PERIOD + 1)
    
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
        if atr_pct < ATR_MIN_PCT or atr_pct > ATR_MAX_PCT:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Calculate EMA differences (trend strength)
        fast_med_diff = (ema_fast[i] - ema_med[i]) / close[i]
        med_slow_diff = (ema_med[i] - ema_slow[i]) / close[i]
        
        # Determine trend direction
        # Both EMA pairs should agree for strong trend
        if fast_med_diff > TREND_MIN_STRENGTH and med_slow_diff > TREND_MIN_STRENGTH:
            trend_dir = 1  # Bullish
        elif fast_med_diff < -TREND_MIN_STRENGTH and med_slow_diff < -TREND_MIN_STRENGTH:
            trend_dir = -1  # Bearish
        else:
            # No clear trend
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Major trend filter (price vs 100 EMA)
        major_trend = np.sign(close[i] - ema_slow[i])
        
        # Only trade in direction of major trend
        if trend_dir != major_trend:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # RSI momentum confirmation
        rsi_factor = 1.0
        if trend_dir > 0:  # Long
            if rsi[i] < RSI_LONG_MIN:
                # RSI too weak for long
                signals[i] = 0.0
                prev_signal = 0.0
                prev_direction = 0
                continue
            elif rsi[i] > 75:
                # Overbought - reduce strength
                rsi_factor = 0.5
        else:  # Short
            if rsi[i] > RSI_SHORT_MAX:
                # RSI too strong for short
                signals[i] = 0.0
                prev_signal = 0.0
                prev_direction = 0
                continue
            elif rsi[i] < 25:
                # Oversold - reduce strength
                rsi_factor = 0.5
        
        # Calculate trend strength
        trend_strength = (abs(fast_med_diff) + abs(med_slow_diff)) / 2
        trend_strength = min(trend_strength * 200, 1.0)  # Scale to 0-1
        
        # Calculate raw signal
        raw_signal = trend_dir * trend_strength * rsi_factor
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
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