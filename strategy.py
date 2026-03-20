#!/usr/bin/env python3
"""
strategy.py - ROC RSI Volume Momentum V1
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Volume-confirmed momentum with RSI timing on 1h timeframe:
    - Primary signal: ROC(10) for momentum direction and strength
    - Entry timing: RSI(14) in neutral zone (40-60) for entry, avoid extremes
    - Trend filter: Price above/below 200 EMA for directional bias
    - Volume confirmation: Volume > 1.3x 20-bar average for breakout validity
    - Volatility filter: ATR normalization for consistent risk
    
    Why this differs from failed macd_rsi_momentum_1h_v1:
    - ROC is more responsive than MACD for crypto momentum
    - Volume confirmation reduces false breakouts
    - RSI neutral zone entry (not extreme) captures momentum continuation
    - Conservative leverage (1.5x) and tighter risk controls
    
    Expected improvement:
    - Better trade timing reduces drawdown vs pure supertrend
    - Volume filter increases win rate on breakouts
    - Should generate 10+ trades with Sharpe > 0.253 (current best)

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

name = "roc_rsi_volume_momentum_v1"
timeframe = "1h"
leverage = 1.5  # Conservative to control drawdown

# Momentum configuration
ROC_PERIOD = 10
ROC_MIN_ABS = 0.015  # Minimum 1.5% momentum to trade

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_MIN = 40  # RSI must be above this for longs
RSI_LONG_MAX = 65  # RSI must be below this (avoid overbought)
RSI_SHORT_MIN = 35  # RSI must be above this (avoid oversold)
RSI_SHORT_MAX = 60  # RSI must be below this for shorts

# Trend filter configuration
EMA_MAJOR = 200
EMA_FAST = 21  # For additional trend confirmation

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 1.30  # Volume must be at least 1.3x average

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_MIN = 0.003  # Minimum ATR % to trade
VOLATILITY_MAX = 0.060  # Maximum ATR % to trade
VOLATILITY_TARGET = 0.020  # Target ATR as % of price

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SMOOTHING_FACTOR = 0.40  # EMA smoothing for signals
HYSTERESIS_THRESHOLD = 0.12  # Minimum change to flip signal direction

# ADX configuration for trend strength
ADX_PERIOD = 14
ADX_MIN = 22  # Minimum ADX to trade (trend strength)


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


def calculate_roc(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Rate of Change using only past data.
    ROC = (close - close_n_periods_ago) / close_n_periods_ago
    """
    n = len(close)
    roc = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return roc
    
    for i in range(period, n):
        if close[i - period] > 0:
            roc[i] = (close[i] - close[i - period]) / close[i - period]
        else:
            roc[i] = 0.0
    
    return roc


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


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average Directional Index using only past data.
    Measures trend strength (not direction).
    """
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth +DM, -DM, and TR
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    tr_series = pd.Series(tr)
    
    smoothed_plus_dm = plus_dm_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    smoothed_minus_dm = minus_dm_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    smoothed_tr = tr_series.ewm(span=period, adjust=False, min_periods=period).mean().values
    
    # Calculate +DI and -DI
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if smoothed_tr[i] > 0:
            plus_di[i] = 100.0 * smoothed_plus_dm[i] / smoothed_tr[i]
            minus_di[i] = 100.0 * smoothed_minus_dm[i] / smoothed_tr[i]
    
    # Calculate DX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Calculate ADX (smoothed DX)
    dx_series = pd.Series(dx)
    adx_series = dx_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    adx = np.nan_to_num(adx_series.values, nan=0.0)
    
    return adx


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    ROC RSI Volume Momentum V1 Strategy.
    
    Signal Logic:
    1. Calculate ROC(10) for momentum direction and strength
    2. Calculate RSI(14) for entry timing (neutral zone preferred)
    3. Filter by 200 EMA trend direction
    4. Confirm with volume > 1.3x average
    5. Filter by ADX > 22 for trend strength
    6. Apply volatility normalization
    7. Smooth signals and apply hysteresis
    
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
    ema_major = calculate_ema(close, EMA_MAJOR)
    ema_fast = calculate_ema(close, EMA_FAST)
    
    roc = calculate_roc(close, ROC_PERIOD)
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_FAST,
        ROC_PERIOD + 1,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        ADX_PERIOD * 2 + 1,
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
        
        # Volatility filter (not too low, not too high)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volume filter (ensure sufficient liquidity and breakout confirmation)
        if volume_ratio[i] < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # ADX filter (ensure trend strength)
        if adx[i] < ADX_MIN:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Momentum signal from ROC
        roc_value = roc[i]
        if abs(roc_value) < ROC_MIN_ABS:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Determine momentum direction
        momentum_direction = np.sign(roc_value)
        momentum_strength = min(1.0, abs(roc_value) / 0.05)  # Normalize to 0-1
        
        # Trend filter (price vs 200 EMA)
        if ema_major[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        trend_direction = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend
        if momentum_direction != trend_direction:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # RSI entry timing filter
        rsi_value = rsi[i]
        rsi_factor = 1.0
        
        if momentum_direction > 0:
            # Long: RSI should be in favorable zone (40-65)
            if rsi_value < RSI_LONG_MIN or rsi_value > RSI_LONG_MAX:
                signals[i] = 0.0
                prev_signal = 0.0
                prev_direction = 0
                continue
            # Better RSI (50-60) gets higher factor
            if 50 <= rsi_value <= 60:
                rsi_factor = 1.0
            elif 40 <= rsi_value < 50:
                rsi_factor = 0.7
            else:  # 60-65
                rsi_factor = 0.6
        else:
            # Short: RSI should be in favorable zone (35-60)
            if rsi_value < RSI_SHORT_MIN or rsi_value > RSI_SHORT_MAX:
                signals[i] = 0.0
                prev_signal = 0.0
                prev_direction = 0
                continue
            # Better RSI (40-50) gets higher factor
            if 40 <= rsi_value <= 50:
                rsi_factor = 1.0
            elif 50 < rsi_value <= 60:
                rsi_factor = 0.7
            else:  # 35-40
                rsi_factor = 0.6
        
        # Additional trend confirmation (fast EMA)
        ema_confirmation = np.sign(ema_fast[i] - ema_major[i])
        if ema_confirmation != trend_direction:
            # Conflicting EMA signals → reduce strength
            momentum_strength *= 0.6
        
        # Calculate raw signal
        raw_signal = momentum_direction * momentum_strength * rsi_factor
        
        # Volatility normalization (scale by target volatility)
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