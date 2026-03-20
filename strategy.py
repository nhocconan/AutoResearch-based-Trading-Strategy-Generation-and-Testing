#!/usr/bin/env python3
"""
strategy.py - Momentum Trend Rider V20
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Pure momentum trend-following with funding rate confirmation:
    - Primary signal: EMA ribbon alignment (12/26/50 all aligned)
    - Confirmation: Funding rate confirms trend direction (not contrarian)
    - Entry timing: RSI momentum (avoid extreme overbought/oversold)
    - Simplified filtering: Less conditions to allow more trades
    - Why this works: Crypto trends persist, funding shows institutional flow
    
    Changes from V12:
    - Funding rate now CONFIRMS trend (not contrarian)
    - Simpler EMA ribbon vs complex crossover logic
    - Reduced hysteresis for faster signal changes
    - Lower minimum signal threshold for more trades
    - Removed volume filter (always sufficient in crypto futures)

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

name = "momentum_trend_rider_v20"
timeframe = "1h"
leverage = 2.5  # Moderate leverage for momentum strategy

# EMA ribbon configuration for trend detection
EMA_FAST = 12
EMA_MID = 26
EMA_SLOW = 50

# RSI configuration for momentum timing
RSI_PERIOD = 14
RSI_LONG_MIN = 45  # Minimum RSI for long entries
RSI_SHORT_MAX = 55  # Maximum RSI for short entries
RSI_EXIT_LONG = 75  # Exit longs when RSI too high
RSI_EXIT_SHORT = 25  # Exit shorts when RSI too low

# Funding rate configuration (trend confirmation, not contrarian)
FUNDING_POSITIVE_THRESHOLD = 0.0005  # Positive funding confirms uptrend
FUNDING_NEGATIVE_THRESHOLD = -0.0005  # Negative funding confirms downtrend
FUNDING_STRENGTH_WEIGHT = 0.25  # How much funding affects signal

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.020  # Target ATR as % of price
VOLATILITY_MIN = 0.002  # Minimum ATR % to trade
VOLATILITY_MAX = 0.080  # Maximum ATR % to trade

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.10  # Lower threshold for more trades
MAX_SIGNAL = 0.90  # Maximum signal magnitude
SMOOTHING_FACTOR = 0.30  # Less smoothing for faster response
TREND_STRENGTH_MIN = 0.15  # Minimum trend strength to trade


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


def calculate_trend_strength(ema_fast: np.ndarray, 
                             ema_mid: np.ndarray,
                             ema_slow: np.ndarray,
                             close: np.ndarray) -> np.ndarray:
    """
    Calculate trend strength from EMA ribbon alignment.
    Returns value in [0, 1] where 1 = perfectly aligned trend.
    Only uses current/past data (no look-ahead).
    """
    n = len(close)
    strength = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if close[i] <= 0:
            strength[i] = 0.0
            continue
        
        # Calculate normalized spreads between EMAs
        fast_mid_spread = (ema_fast[i] - ema_mid[i]) / close[i]
        mid_slow_spread = (ema_mid[i] - ema_slow[i]) / close[i]
        
        # Bullish alignment: fast > mid > slow
        bullish = fast_mid_spread > 0 and mid_slow_spread > 0
        # Bearish alignment: fast < mid < slow
        bearish = fast_mid_spread < 0 and mid_slow_spread < 0
        
        if bullish:
            # Strength based on spread magnitude
            strength[i] = min(1.0, (fast_mid_spread + mid_slow_spread) * 100)
        elif bearish:
            # Strength based on spread magnitude (negative trend)
            strength[i] = min(1.0, (abs(fast_mid_spread) + abs(mid_slow_spread)) * 100)
        else:
            # EMAs not aligned - weak trend
            strength[i] = 0.0
    
    return strength


def calculate_funding_confirmation(funding_rate: np.ndarray,
                                   positive_threshold: float = 0.0005,
                                   negative_threshold: float = -0.0005,
                                   weight: float = 0.25) -> np.ndarray:
    """
    Calculate funding rate as trend confirmation (not contrarian).
    Positive funding + uptrend = stronger long signal
    Negative funding + downtrend = stronger short signal
    Returns value in [-weight, weight].
    Only uses current/past funding rate (no look-ahead).
    """
    n = len(funding_rate)
    signal = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        fr = funding_rate[i]
        
        if fr > positive_threshold:
            # Positive funding confirms bullish sentiment
            signal[i] = weight * min(1.0, fr / positive_threshold)
        elif fr < negative_threshold:
            # Negative funding confirms bearish sentiment
            signal[i] = weight * min(1.0, fr / negative_threshold)
        else:
            # Neutral funding
            signal[i] = weight * (fr / positive_threshold) if positive_threshold != 0 else 0.0
    
    return signal


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Momentum Trend Rider V20 Strategy.
    
    Signal Logic:
    1. Calculate EMA ribbon (12/26/50) for trend direction and strength
    2. Calculate funding rate confirmation (supports trend direction)
    3. Calculate RSI for momentum timing
    4. Combine signals: trend strength + funding confirmation
    5. Apply RSI filters for entry/exit timing
    6. Apply volatility normalization
    7. Smooth signals lightly for stability
    8. Apply minimum magnitude filter
    
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
    ema_mid = calculate_ema(close, EMA_MID)
    ema_slow = calculate_ema(close, EMA_SLOW)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    trend_strength = calculate_trend_strength(ema_fast, ema_mid, ema_slow, close)
    funding_signal = calculate_funding_confirmation(
        funding_rate,
        FUNDING_POSITIVE_THRESHOLD,
        FUNDING_NEGATIVE_THRESHOLD,
        FUNDING_STRENGTH_WEIGHT
    )
    
    # Determine trend direction from EMA ribbon
    trend_direction = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if ema_fast[i] > ema_mid[i] and ema_mid[i] > ema_slow[i]:
            trend_direction[i] = 1.0  # Bullish
        elif ema_fast[i] < ema_mid[i] and ema_mid[i] < ema_slow[i]:
            trend_direction[i] = -1.0  # Bearish
        else:
            trend_direction[i] = 0.0  # Neutral/unclear
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1
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
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Check trend strength minimum
        if trend_strength[i] < TREND_STRENGTH_MIN:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Get trend direction
        direction = trend_direction[i]
        
        # RSI timing filter
        rsi_valid = True
        if direction > 0:
            # Long: RSI should be in momentum zone, not overbought
            if rsi[i] < RSI_LONG_MIN or rsi[i] > RSI_EXIT_LONG:
                rsi_valid = False
        elif direction < 0:
            # Short: RSI should be in momentum zone, not oversold
            if rsi[i] > RSI_SHORT_MAX or rsi[i] < RSI_EXIT_SHORT:
                rsi_valid = False
        
        if not rsi_valid:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate base signal from trend strength and direction
        base_signal = direction * trend_strength[i]
        
        # Add funding confirmation
        # Funding should align with trend direction for confirmation
        if direction > 0 and funding_signal[i] > 0:
            # Bullish trend + positive funding = reinforced long
            combined_signal = base_signal + funding_signal[i]
        elif direction < 0 and funding_signal[i] < 0:
            # Bearish trend + negative funding = reinforced short
            combined_signal = base_signal + funding_signal[i]
        else:
            # Funding neutral or conflicting - reduce signal slightly
            combined_signal = base_signal * 0.8
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        raw_signal = combined_signal * vol_factor
        
        # Signal smoothing (light EMA on signals)
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals