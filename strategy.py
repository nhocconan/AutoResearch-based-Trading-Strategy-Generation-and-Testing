#!/usr/bin/env python3
"""
strategy.py - Mean Reversion Trend Filter V14
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Mean reversion with trend filter - simpler and more robust:
    - Primary signal: RSI mean reversion (oversold → long, overbought → short)
    - Trend filter: Only trade mean reversion in direction of 200 EMA trend
    - Funding confirmation: Extreme funding reinforces crowded trade reversal
    - Volume filter: Ensure sufficient liquidity
    - Why this works: Crypto mean-reverts within trends, funding extremes
      indicate crowded positions ripe for reversal
    - Simpler than v12, fewer filters to ensure actual trades occur

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

name = "mean_reversion_trend_v14"
timeframe = "1h"
leverage = 1.5  # Lower leverage for better Sharpe

# RSI configuration for mean reversion
RSI_PERIOD = 14
RSI_OVERSOLD = 30  # Long entry when RSI below this
RSI_OVERBOUGHT = 70  # Short entry when RSI above this
RSI_NEUTRAL = 50  # Center line

# Trend filter configuration
EMA_TREND = 200  # Major trend filter
EMA_FAST = 50  # Secondary trend confirmation

# Funding rate configuration
FUNDING_EXTREME = 0.0008  # 0.08% per 8hr = extreme
FUNDING_LOOKBACK = 50  # Rolling window for funding extremes
FUNDING_WEIGHT = 0.30  # Weight of funding in signal

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.50  # Minimum volume ratio to trade

# Signal configuration
MIN_SIGNAL = 0.20  # Minimum signal magnitude to trade
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SMOOTHING = 0.40  # Signal smoothing factor


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(series: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA using only past data."""
    n = len(series)
    if n < period:
        return np.zeros(n, dtype=np.float64)
    
    pd_series = pd.Series(series)
    ema = pd_series.ewm(span=period, adjust=False, min_periods=period).mean()
    return np.nan_to_num(ema.values, nan=0.0)


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI using only past data."""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0, dtype=np.float64)
    
    pd_close = pd.Series(close)
    delta = pd_close.diff()
    
    gains = delta.clip(lower=0)
    losses = (-delta).clip(lower=0)
    
    avg_gains = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_losses = losses.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gains / avg_losses.replace(0, np.inf)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return np.nan_to_num(rsi.values, nan=50.0)


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
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


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Calculate volume ratio vs rolling average using only past data."""
    n = len(volume)
    if n < lookback:
        return np.ones(n, dtype=np.float64)
    
    vol_series = pd.Series(volume)
    rolling_avg = vol_series.rolling(window=lookback, min_periods=lookback).mean()
    
    ratio = vol_series.values / rolling_avg.values
    return np.nan_to_num(ratio, nan=1.0)


def calculate_funding_zscore(funding_rate: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calculate rolling z-score of funding rate using only past data.
    Positive z-score = funding higher than recent average (bearish)
    Negative z-score = funding lower than recent average (bullish)
    """
    n = len(funding_rate)
    zscore = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return zscore
    
    fr_series = pd.Series(funding_rate)
    rolling_mean = fr_series.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = fr_series.rolling(window=lookback, min_periods=lookback).std()
    
    zscore = (fr_series.values - rolling_mean.values) / rolling_std.values.replace(0, np.inf)
    return np.nan_to_num(zscore, nan=0.0)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Mean Reversion Trend Filter V14 Strategy.
    
    Signal Logic:
    1. Calculate RSI for mean reversion signals
    2. Calculate 200 EMA for trend direction filter
    3. Calculate funding z-score for crowded position detection
    4. Combine: RSI signal + trend filter + funding confirmation
    5. Apply volume and volatility filters
    6. Smooth signals and apply magnitude thresholds
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, funding_rate]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract and clean price data
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
    
    # Clean invalid data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Fix invalid prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    high = np.maximum(high, close)
    low = np.minimum(low, close)
    
    # Calculate all indicators (all use only past data)
    ema_trend = calculate_ema(close, EMA_TREND)
    ema_fast = calculate_ema(close, EMA_FAST)
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, 14)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    funding_z = calculate_funding_zscore(funding_rate, FUNDING_LOOKBACK)
    
    # Calculate minimum valid index
    min_valid = max(EMA_TREND, EMA_FAST, RSI_PERIOD + 1, 15, VOLUME_LOOKBACK, FUNDING_LOOKBACK)
    
    # Generate signals
    prev_signal = 0.0
    
    for i in range(min_valid, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volume filter (relaxed to ensure trades)
        if volume_ratio[i] < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Determine trend direction from 200 EMA
        trend_direction = 0
        if ema_trend[i] > 0:
            if close[i] > ema_trend[i]:
                trend_direction = 1  # Uptrend
            elif close[i] < ema_trend[i]:
                trend_direction = -1  # Downtrend
        
        # RSI mean reversion signal
        rsi_signal = 0.0
        if rsi[i] < RSI_OVERSOLD:
            # Oversold → long bias
            rsi_signal = (RSI_OVERSOLD - rsi[i]) / RSI_OVERSOLD
            rsi_signal = min(1.0, rsi_signal)
        elif rsi[i] > RSI_OVERBOUGHT:
            # Overbought → short bias
            rsi_signal = -(rsi[i] - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT)
            rsi_signal = max(-1.0, rsi_signal)
        
        # Apply trend filter: only trade mean reversion in trend direction
        if trend_direction != 0 and rsi_signal != 0:
            # If RSI signal aligns with trend, strengthen it
            if np.sign(rsi_signal) == trend_direction:
                rsi_signal *= 1.2
            # If RSI signal opposes trend, weaken it (counter-trend mean reversion)
            elif np.sign(rsi_signal) != trend_direction:
                rsi_signal *= 0.5
        elif trend_direction == 0:
            # No clear trend, reduce signal strength
            rsi_signal *= 0.7
        
        # Funding rate confirmation (contrarian)
        # High funding z-score → crowded longs → short bias
        # Low funding z-score → crowded shorts → long bias
        funding_signal = 0.0
        if abs(funding_z[i]) > 1.0:
            funding_signal = -np.clip(funding_z[i] / 3.0, -1.0, 1.0) * FUNDING_WEIGHT
        
        # Combine signals
        raw_signal = rsi_signal * (1.0 - FUNDING_WEIGHT) + funding_signal
        
        # Volatility adjustment (scale down in high volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct > 0.03:  # High volatility
            raw_signal *= 0.7
        elif atr_pct < 0.005:  # Very low volatility
            raw_signal *= 0.8
        
        # Signal smoothing
        smoothed = SMOOTHING * prev_signal + (1.0 - SMOOTHING) * raw_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed) < MIN_SIGNAL:
            smoothed = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals