#!/usr/bin/env python3
"""
strategy.py - Multi Trend 4H V1
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "4h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Multi-timeframe trend following on 4h chart:
    - Primary signal: EMA crossover (13/34) for trend direction
    - Trend filter: Price above/below 100 EMA confirms major trend
    - Momentum: RSI between 35-65 avoids overbought/oversold entries
    - Volatility: ATR normalization for consistent risk across assets
    - Funding: Extreme funding rates provide contrarian overlay
    
    Why 4h timeframe:
    - Fewer whipsaws than 1h/15m
    - More trades than daily
    - Cleaner trend signals for crypto
    - Lower transaction cost impact vs lower timeframes

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

name = "multi_trend_4h_v1"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for controlled drawdown

# EMA configuration for trend detection
EMA_FAST = 13
EMA_SLOW = 34
EMA_MAJOR = 100

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_MIN = 35  # RSI must be above this for longs
RSI_LONG_MAX = 65  # RSI must be below this for longs
RSI_SHORT_MIN = 35  # RSI must be above this for shorts
RSI_SHORT_MAX = 65  # RSI must be below this for shorts

# Funding rate configuration
FUNDING_EXTREME = 0.0008  # 0.08% per 8hr = extreme
FUNDING_LOOKBACK = 50  # For calculating extremes
FUNDING_WEIGHT = 0.25  # How much funding affects signal

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.020  # Target ATR as % of price
VOLATILITY_MIN = 0.005  # Minimum ATR % to trade
VOLATILITY_MAX = 0.080  # Maximum ATR % to trade

# Signal configuration
MIN_SIGNAL = 0.20  # Minimum signal magnitude to trade
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.40  # EMA smoothing factor for signals

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.60  # Volume must be at least this % of average


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


def calculate_funding_zscore(funding_rate: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calculate rolling z-score of funding rate.
    Positive z-score = funding above average (bearish signal)
    Negative z-score = funding below average (bullish signal)
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    zscore = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return zscore
    
    funding_series = pd.Series(funding_rate)
    rolling_mean = funding_series.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = funding_series.rolling(window=lookback, min_periods=lookback).std()
    
    zscore = np.nan_to_num((funding_series.values - rolling_mean.values) / rolling_std.values, nan=0.0)
    
    return zscore


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Multi Trend 4H V1 Strategy.
    
    Signal Logic:
    1. Calculate EMA crossover signal (13/34 EMA)
    2. Filter by major trend (price vs 100 EMA)
    3. Confirm with RSI momentum (avoid extremes)
    4. Apply funding rate z-score overlay (contrarian)
    5. Normalize by volatility (ATR)
    6. Smooth signals and apply minimum magnitude filter
    
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
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    funding_zscore = calculate_funding_zscore(funding_rate, FUNDING_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        FUNDING_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0 or ema_major[i] <= 0:
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
        
        # Determine trend direction from EMA crossover
        ema_diff = ema_fast[i] - ema_slow[i]
        ema_direction = np.sign(ema_diff)
        
        # Major trend filter (price vs 100 EMA)
        major_trend = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend
        if ema_direction != major_trend:
            # Conflicting signals → no trade
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # RSI momentum confirmation
        rsi_valid = False
        if ema_direction > 0:
            # Long: RSI between 35-65 (not overbought)
            if RSI_LONG_MIN <= rsi[i] <= RSI_LONG_MAX:
                rsi_valid = True
        elif ema_direction < 0:
            # Short: RSI between 35-65 (not oversold)
            if RSI_SHORT_MIN <= rsi[i] <= RSI_SHORT_MAX:
                rsi_valid = True
        
        if not rsi_valid:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate base signal strength from EMA separation
        ema_strength = abs(ema_diff) / close[i] * 100  # Normalize by price
        ema_strength = np.clip(ema_strength, 0, 1.0)
        
        # Base signal with trend direction
        raw_signal = ema_direction * ema_strength
        
        # Funding rate overlay (contrarian)
        # High positive funding z-score → reduce long / add short bias
        # High negative funding z-score → reduce short / add long bias
        funding_overlay = -funding_zscore[i] * FUNDING_WEIGHT * 0.5
        funding_overlay = np.clip(funding_overlay, -FUNDING_WEIGHT, FUNDING_WEIGHT)
        
        # Combine signals
        raw_signal = raw_signal * (1.0 - FUNDING_WEIGHT) + funding_overlay
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
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
    
    return signals