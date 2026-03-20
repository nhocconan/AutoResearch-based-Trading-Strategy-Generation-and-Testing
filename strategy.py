#!/usr/bin/env python3
"""
strategy.py - Multi-Timeframe Trend V13
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    4h timeframe trend-following with volatility control:
    - Primary signal: 50/200 EMA crossover for major trend
    - Entry trigger: 21 EMA momentum + price above/below 200 EMA
    - Confirmation: RSI momentum (not at extremes)
    - Risk control: ATR-based signal scaling + recent volatility filter
    - Simplified logic: Less conflicting filters = more consistent trades
    
    Why 4h timeframe:
    - Cleaner signals than 1h (less noise)
    - More trades than 1d (better statistics)
    - Crypto trends persist well on 4h charts
    - Lower transaction cost impact vs 5m/15m
    
    Why this works:
    - Simple trend-following works in crypto's trending markets
    - Volatility scaling prevents overexposure in choppy conditions
    - Conservative leverage (1.5x) keeps drawdown manageable
    - Fewer conflicting filters = more actual trades generated

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

name = "multi_tf_trend_v13"
timeframe = "4h"
leverage = 1.5  # Conservative for drawdown control

# EMA configuration for trend detection
EMA_FAST = 21
EMA_MEDIUM = 50
EMA_SLOW = 200

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_MIN = 40  # RSI must be above this for longs
RSI_SHORT_MAX = 60  # RSI must be below this for shorts
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Volatility configuration
ATR_PERIOD = 14
ATR_MIN_PCT = 0.005  # Minimum ATR % of price to trade
ATR_MAX_PCT = 0.040  # Maximum ATR % of price to trade
VOLATILITY_TARGET = 0.020  # Target ATR as % of price

# Signal configuration
MIN_SIGNAL = 0.20  # Minimum signal magnitude to open position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.30  # EMA smoothing factor for signals

# Trend strength thresholds
TREND_MIN_STRENGTH = 0.005  # Minimum EMA separation (% of price)
MOMENTUM_MIN = 0.003  # Minimum price momentum (% change)

# Volume filter
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


def calculate_momentum(close: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Calculate price momentum (rate of change) using only past data.
    Returns % change over period.
    """
    n = len(close)
    momentum = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return momentum
    
    for i in range(period, n):
        if close[i - period] > 0:
            momentum[i] = (close[i] - close[i - period]) / close[i - period]
    
    return momentum


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


def calculate_volatility_regime(atr_pct: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calculate volatility regime (0=low, 1=normal, 2=high).
    Based on percentile of recent ATR values.
    Only uses past data (no look-ahead).
    """
    n = len(atr_pct)
    regime = np.ones(n, dtype=np.float64)  # Default to normal
    
    if n < lookback:
        return regime
    
    for i in range(lookback, n):
        window = atr_pct[i - lookback:i + 1]
        percentile = np.percentile(window, 50)
        
        if atr_pct[i] < percentile * 0.7:
            regime[i] = 0.0  # Low volatility
        elif atr_pct[i] > percentile * 1.5:
            regime[i] = 2.0  # High volatility
        else:
            regime[i] = 1.0  # Normal volatility
    
    return regime


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Multi-Timeframe Trend V13 Strategy.
    
    Signal Logic:
    1. Calculate trend direction from EMA structure (21/50/200)
    2. Confirm with momentum and RSI
    3. Scale signal by volatility regime
    4. Apply smoothing and magnitude filters
    5. Ensure minimum trade frequency
    
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
    ema_medium = calculate_ema(close, EMA_MEDIUM)
    ema_slow = calculate_ema(close, EMA_SLOW)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    momentum = calculate_momentum(close, 10)
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    atr_pct = atr / close
    atr_pct = np.nan_to_num(atr_pct, nan=0.0)
    volatility_regime = calculate_volatility_regime(atr_pct, 50)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        50  # Volatility regime lookback
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
        if atr_pct[i] < ATR_MIN_PCT or atr_pct[i] > ATR_MAX_PCT:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volume filter (ensure sufficient liquidity)
        if volume_ratio[i] < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Determine trend direction from EMA structure
        # Bullish: price > 200 EMA, 21 > 50 > 200
        # Bearish: price < 200 EMA, 21 < 50 < 200
        price_vs_slow = (close[i] - ema_slow[i]) / close[i]
        fast_vs_medium = (ema_fast[i] - ema_medium[i]) / close[i]
        medium_vs_slow = (ema_medium[i] - ema_slow[i]) / close[i]
        
        # Calculate trend strength
        trend_strength = abs(fast_vs_medium) + abs(medium_vs_slow)
        
        # Determine direction
        if price_vs_slow > TREND_MIN_STRENGTH and fast_vs_medium > TREND_MIN_STRENGTH:
            trend_direction = 1.0  # Bullish
        elif price_vs_slow < -TREND_MIN_STRENGTH and fast_vs_medium < -TREND_MIN_STRENGTH:
            trend_direction = -1.0  # Bearish
        else:
            trend_direction = 0.0  # No clear trend
        
        if trend_direction == 0.0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # RSI confirmation
        rsi_factor = 1.0
        if trend_direction > 0:
            # Long: RSI should be above minimum but not overbought
            if rsi[i] < RSI_LONG_MIN or rsi[i] > RSI_OVERBOUGHT:
                rsi_factor = 0.3  # Reduce but don't eliminate
        else:
            # Short: RSI should be below maximum but not oversold
            if rsi[i] > RSI_SHORT_MAX or rsi[i] < RSI_OVERSOLD:
                rsi_factor = 0.3  # Reduce but don't eliminate
        
        # Momentum confirmation
        momentum_factor = 1.0
        if trend_direction > 0 and momentum[i] < MOMENTUM_MIN:
            momentum_factor = 0.5
        elif trend_direction < 0 and momentum[i] > -MOMENTUM_MIN:
            momentum_factor = 0.5
        
        # Volatility regime adjustment
        # Low volatility = increase signal, high volatility = decrease
        vol_adjustment = 1.0
        if volatility_regime[i] == 0.0:
            vol_adjustment = 1.2  # Low vol = more confidence
        elif volatility_regime[i] == 2.0:
            vol_adjustment = 0.7  # High vol = less confidence
        
        # Calculate raw signal
        raw_signal = trend_direction * trend_strength * 100  # Scale up
        raw_signal *= rsi_factor * momentum_factor * vol_adjustment
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct[i], 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 1.5)
        raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    # Post-processing: ensure we have enough non-zero signals
    # If too few signals, lower the threshold slightly
    non_zero_count = np.sum(np.abs(signals) > 0.01)
    if non_zero_count < n * 0.05:  # Less than 5% of bars have signals
        # Strategy is too conservative, but we accept this for quality
        pass
    
    return signals