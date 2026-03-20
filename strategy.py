#!/usr/bin/env python3
"""
strategy.py - Trend Volatility Controlled V8
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "4h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    4h timeframe trend-following with volatility-based position sizing:
    - Primary signal: EMA crossover (12/26) for trend direction
    - Confirmation: ADX > 25 for trend strength validation
    - Filter: Price above/below 200 EMA for major trend alignment
    - Risk control: Volatility scaling reduces position size in high vol
    - Entry timing: RSI not at extremes (30-70 range preferred)
    
    Why 4h timeframe:
    - Cleaner signals than 1h/15m, less noise
    - More trades than 1d, better statistical significance
    - Lower transaction cost impact than lower timeframes
    - Works well across BTC/ETH/SOL (tested concept)
    
    Why this should control drawdown:
    - Volatility scaling reduces exposure when ATR is high
    - ADX filter avoids trading in choppy/ranging markets
    - Conservative leverage (1.5x max)
    - 200 EMA filter prevents counter-trend trades

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

name = "trend_volatility_controlled_v8"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for drawdown control

# EMA configuration for trend detection
EMA_FAST = 12
EMA_SLOW = 26
EMA_MAJOR = 200

# ADX configuration for trend strength
ADX_PERIOD = 14
ADX_THRESHOLD = 25  # Minimum ADX to consider trend valid

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_MIN = 35  # Don't long if RSI below this
RSI_MAX = 65  # Don't short if RSI above this

# Volatility configuration for position sizing
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.02  # Target ATR as % of price
VOLATILITY_MIN = 0.005  # Minimum ATR % to trade
VOLATILITY_MAX = 0.08  # Maximum ATR % (reduce position above this)

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.3  # EMA smoothing factor for signals

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.5  # Volume must be at least this % of average


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


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average Directional Index using only past data.
    ADX measures trend strength (not direction).
    """
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    
    if n < period * 2 + 1:
        return adx
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Calculate ATR for TR
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Smooth DM and TR
    tr_series = pd.Series(tr)
    atr_series = tr_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    
    plus_di = np.nan_to_num(
        (plus_dm_series.ewm(span=period, adjust=False, min_periods=period).mean().values / 
         atr_series.values) * 100,
        nan=0.0
    )
    minus_di = np.nan_to_num(
        (minus_dm_series.ewm(span=period, adjust=False, min_periods=period).mean().values / 
         atr_series.values) * 100,
        nan=0.0
    )
    
    # Calculate DX and ADX
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)
    
    dx = np.where(di_sum > 0, (di_diff / di_sum) * 100, 0.0)
    
    dx_series = pd.Series(dx)
    adx_series = dx_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    adx = np.nan_to_num(adx_series.values, nan=0.0)
    
    return adx


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


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Volatility Controlled V8 Strategy.
    
    Signal Logic:
    1. Calculate trend signal from EMA crossover (12/26)
    2. Confirm trend strength with ADX > 25
    3. Filter by major trend (200 EMA)
    4. Check RSI not at extremes
    5. Scale signal by volatility (reduce size in high vol)
    6. Smooth signals to reduce whipsaws
    7. Apply minimum magnitude filter
    
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
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW + EMA_SLOW,  # ADX needs extra warmup
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK
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
        
        # Volume filter (ensure sufficient liquidity)
        if volume_ratio[i] < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # ADX trend strength filter
        if adx[i] < ADX_THRESHOLD:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Determine trend direction from EMA crossover
        ema_diff = ema_fast[i] - ema_slow[i]
        ema_direction = np.sign(ema_diff)
        
        # Major trend filter (price vs 200 EMA)
        if ema_major[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        major_filter = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend
        if ema_direction != major_filter:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # RSI filter (avoid extremes)
        if ema_direction > 0 and rsi[i] < RSI_MIN:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        if ema_direction < 0 and rsi[i] > RSI_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate base signal strength from EMA diff
        ema_strength = abs(ema_diff) / close[i] * 100  # Normalize as percentage
        ema_strength = np.clip(ema_strength, 0, 10)  # Cap at 10%
        
        # Normalize to 0-1 range
        trend_signal = ema_direction * (ema_strength / 10.0)
        
        # Volatility scaling (reduce position size when volatility is high)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        
        # Apply volatility scaling
        raw_signal = trend_signal * vol_factor
        
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