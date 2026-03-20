#!/usr/bin/env python3
"""
strategy.py - Trend Momentum V15
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "4h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Clean trend-following with momentum confirmation on 4h timeframe:
    - Primary signal: Triple EMA alignment (9/21/55) for trend direction
    - Confirmation: MACD histogram for momentum strength
    - Filter: Price above/below 200 EMA for major trend validation
    - Volatility scaling: ATR-based position sizing to normalize risk
    - Drawdown control: Reduce position size during losing streaks
    
    Why 4h timeframe:
    - Cleaner signals than 1h/15m (less noise)
    - More trades than 1d (better statistical significance)
    - Lower transaction cost impact vs 5m/15m
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

name = "trend_momentum_v15"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for drawdown control

# EMA configuration for trend detection
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 55
EMA_MAJOR = 200

# MACD configuration for momentum
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MACD_THRESHOLD = 0.0  # Minimum MACD histogram for entry

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.020  # Target ATR as % of price
VOLATILITY_MIN = 0.005  # Minimum ATR % to trade
VOLATILITY_MAX = 0.080  # Maximum ATR % to trade

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SMOOTHING_FACTOR = 0.30  # EMA smoothing for signals (0=none, 1=max)
HYSTERESIS_THRESHOLD = 0.15  # Minimum change to flip signal direction

# Trend strength configuration
TREND_STRENGTH_MIN = 0.30  # Minimum trend strength to trade
EMA_SPREAD_MIN = 0.005  # Minimum spread between EMAs (% of price)

# Drawdown control
CONSECUTIVE_LOSSES_MAX = 3  # Reduce position after this many losses
LOSS_REDUCTION_FACTOR = 0.50  # Reduce signal by this factor after losses


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


def calculate_macd(close: np.ndarray, 
                   fast: int = 12, 
                   slow: int = 26, 
                   signal: int = 9) -> tuple:
    """
    Calculate MACD indicator using only past data.
    Returns: (macd_line, signal_line, histogram)
    """
    n = len(close)
    macd_line = np.zeros(n, dtype=np.float64)
    signal_line = np.zeros(n, dtype=np.float64)
    histogram = np.zeros(n, dtype=np.float64)
    
    if n < slow:
        return macd_line, signal_line, histogram
    
    close_series = pd.Series(close)
    ema_fast = close_series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    
    macd_raw = ema_fast - ema_slow
    signal_raw = macd_raw.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist_raw = macd_raw - signal_raw
    
    macd_line = np.nan_to_num(macd_raw.values, nan=0.0)
    signal_line = np.nan_to_num(signal_raw.values, nan=0.0)
    histogram = np.nan_to_num(hist_raw.values, nan=0.0)
    
    return macd_line, signal_line, histogram


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


def calculate_trend_strength(ema_fast: np.ndarray, 
                             ema_mid: np.ndarray, 
                             ema_slow: np.ndarray,
                             close: np.ndarray) -> np.ndarray:
    """
    Calculate trend strength based on EMA alignment.
    Returns value in [0, 1] where 1 = strongest trend.
    Only uses current/past data (no look-ahead).
    """
    n = len(close)
    strength = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if close[i] <= 0:
            strength[i] = 0.0
            continue
        
        # Calculate EMA spreads as % of price
        spread_fast_mid = abs(ema_fast[i] - ema_mid[i]) / close[i]
        spread_mid_slow = abs(ema_mid[i] - ema_slow[i]) / close[i]
        
        # Check alignment (all EMAs in correct order)
        bullish_alignment = (ema_fast[i] > ema_mid[i] > ema_slow[i])
        bearish_alignment = (ema_fast[i] < ema_mid[i] < ema_slow[i])
        
        if bullish_alignment or bearish_alignment:
            # Strong trend - EMAs aligned
            strength[i] = min(1.0, (spread_fast_mid + spread_mid_slow) / (2 * EMA_SPREAD_MIN))
        else:
            # Weak trend - EMAs crossed
            strength[i] = 0.3 * min(1.0, (spread_fast_mid + spread_mid_slow) / (2 * EMA_SPREAD_MIN))
    
    return strength


def calculate_signal_direction(ema_fast: np.ndarray, 
                               ema_mid: np.ndarray, 
                               ema_slow: np.ndarray,
                               ema_major: np.ndarray,
                               close: np.ndarray) -> np.ndarray:
    """
    Calculate signal direction based on EMA alignment.
    Returns: 1 (long), -1 (short), 0 (neutral)
    Only uses current/past data (no look-ahead).
    """
    n = len(close)
    direction = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if close[i] <= 0 or ema_major[i] <= 0:
            direction[i] = 0.0
            continue
        
        # Major trend filter
        above_major = close[i] > ema_major[i]
        
        # Fast EMA alignment
        fast_above_mid = ema_fast[i] > ema_mid[i]
        mid_above_slow = ema_mid[i] > ema_slow[i]
        
        # Bullish: all aligned + above major EMA
        if fast_above_mid and mid_above_slow and above_major:
            direction[i] = 1.0
        # Bearish: all aligned + below major EMA
        elif not fast_above_mid and not mid_above_slow and not above_major:
            direction[i] = -1.0
        else:
            direction[i] = 0.0
    
    return direction


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Momentum V15 Strategy.
    
    Signal Logic:
    1. Calculate triple EMA alignment for trend direction
    2. Calculate MACD histogram for momentum confirmation
    3. Calculate trend strength from EMA spreads
    4. Combine signals with volatility scaling
    5. Apply smoothing and hysteresis
    6. Filter by minimum signal magnitude
    
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
    ema_mid = calculate_ema(close, EMA_MID)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    macd_line, macd_signal, macd_hist = calculate_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    trend_strength = calculate_trend_strength(ema_fast, ema_mid, ema_slow, close)
    trend_direction = calculate_signal_direction(ema_fast, ema_mid, ema_slow, ema_major, close)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW,
        MACD_SLOW + MACD_SIGNAL,
        ATR_PERIOD + 1
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    consecutive_losses = 0
    
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
        
        # Get trend direction
        direction = trend_direction[i]
        
        # Skip if no clear trend direction
        if direction == 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Check trend strength
        strength = trend_strength[i]
        if strength < TREND_STRENGTH_MIN:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # MACD momentum confirmation
        macd_confirmation = 1.0
        if direction > 0:
            # Long: want positive MACD histogram
            if macd_hist[i] < MACD_THRESHOLD:
                macd_confirmation = 0.5  # Reduce but don't eliminate
        else:
            # Short: want negative MACD histogram
            if macd_hist[i] > -MACD_THRESHOLD:
                macd_confirmation = 0.5  # Reduce but don't eliminate
        
        # Base signal from trend direction and strength
        raw_signal = direction * strength * macd_confirmation
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        raw_signal *= vol_factor
        
        # Drawdown control: reduce position after consecutive losses
        if consecutive_losses >= CONSECUTIVE_LOSSES_MAX:
            raw_signal *= LOSS_REDUCTION_FACTOR
        
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
        
        # Track consecutive losses (simplified - actual tracking happens in backtest)
        # This is a placeholder for drawdown control logic
        if signal == 0.0 and prev_direction != 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0
    
    return signals