#!/usr/bin/env python3
"""
strategy.py - Trend Funding Simple V16
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simplified trend-following with funding rate overlay:
    - Primary signal: Classic EMA crossover (12/26) for responsiveness
    - Trend filter: Price above/below 100 EMA for direction bias
    - Funding filter: Only act on extreme funding (>0.01% or <-0.01%)
    - Momentum: RSI for entry timing (avoid extremes)
    - Volatility: ATR-based position sizing
    
    Why this should work better:
    - Simpler = less overfitting, more robust
    - Classic EMA periods (12/26) work well in crypto trends
    - Funding only matters at true extremes
    - Less filtering = capture more trending moves
    - Better signal smoothing without excessive lag

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

name = "trend_funding_simple_v16"
timeframe = "1h"
leverage = 2.5  # Moderate leverage for trend following

# EMA configuration - classic MACD-style periods
EMA_FAST = 12
EMA_SLOW = 26
EMA_TREND = 100

# RSI configuration
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_NEUTRAL = 50

# Funding rate configuration - simplified
FUNDING_EXTREME = 0.0001  # 0.01% per 8hr = extreme
FUNDING_WEIGHT = 0.25  # Funding impact on signal

# Volatility configuration
ATR_PERIOD = 14
VOL_TARGET = 0.02  # Target ATR as % of price

# Signal configuration
MIN_SIGNAL = 0.20  # Minimum signal magnitude to trade
MAX_SIGNAL = 0.90  # Maximum signal magnitude
SMOOTHING = 0.3  # Signal EMA smoothing factor


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
    return np.nan_to_num(ema_values, nan=0.0)


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using only past data.
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0, dtype=np.float64)
    
    close_series = pd.Series(close)
    delta = close_series.diff()
    
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    avg_gains = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_losses = losses.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gains / avg_losses.replace(0, np.inf)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    
    return np.nan_to_num(rsi_series.values, nan=50.0)


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
    
    return np.nan_to_num(atr_series.values, nan=0.0)


def calculate_funding_extremes(funding_rate: np.ndarray, lookback: int = 100) -> tuple:
    """
    Calculate rolling percentile extremes of funding rate.
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    if n < lookback:
        return np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
    
    funding_series = pd.Series(funding_rate)
    rolling_high = funding_series.rolling(window=lookback, min_periods=lookback).quantile(0.90)
    rolling_low = funding_series.rolling(window=lookback, min_periods=lookback).quantile(0.10)
    
    return (
        np.nan_to_num(rolling_high.values, nan=0.0),
        np.nan_to_num(rolling_low.values, nan=0.0)
    )


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Funding Simple V16 Strategy.
    
    Signal Logic:
    1. Calculate EMA crossover signal (12/26)
    2. Filter by major trend (price vs 100 EMA)
    3. Apply RSI momentum filter
    4. Add funding rate contrarian overlay at extremes
    5. Normalize by volatility
    6. Smooth signals
    
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
    
    # Clean data - fix invalid prices
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators (all use only past data)
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_trend = calculate_ema(close, EMA_TREND)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    funding_high, funding_low = calculate_funding_extremes(funding_rate, 100)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_TREND,
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        100  # funding lookback
    )
    
    # Generate signals
    prev_signal = 0.0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0 or ema_trend[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate EMA crossover signal
        ema_diff = (ema_fast[i] - ema_slow[i]) / close[i]
        ema_signal = np.sign(ema_diff) * min(1.0, abs(ema_diff) * 100)
        
        # Major trend filter - only trade with the trend
        trend_direction = np.sign(close[i] - ema_trend[i])
        
        # If EMA signal conflicts with major trend, reduce strength
        if np.sign(ema_signal) != trend_direction and trend_direction != 0:
            ema_signal *= 0.5
        
        # RSI momentum filter
        rsi_factor = 1.0
        if ema_signal > 0:
            # Long: avoid overbought
            if rsi[i] > RSI_OVERBOUGHT:
                rsi_factor = 0.5
            elif rsi[i] < 30:
                rsi_factor = 0.3  # Weak long when oversold (might be bottoming)
        elif ema_signal < 0:
            # Short: avoid oversold
            if rsi[i] < RSI_OVERSOLD:
                rsi_factor = 0.5
            elif rsi[i] > 70:
                rsi_factor = 0.3  # Weak short when overbought (might be topping)
        
        # Apply RSI filter
        trend_signal = ema_signal * rsi_factor
        
        # Funding rate contrarian overlay
        funding_signal = 0.0
        fr = funding_rate[i]
        
        # Only act on extreme funding
        if fr > FUNDING_EXTREME:
            # Extreme positive funding = crowded longs = short bias
            funding_signal = -FUNDING_WEIGHT * min(1.0, fr / FUNDING_EXTREME)
        elif fr < -FUNDING_EXTREME:
            # Extreme negative funding = crowded shorts = long bias
            funding_signal = FUNDING_WEIGHT * min(1.0, abs(fr) / FUNDING_EXTREME)
        
        # Combine signals - trend is primary, funding is overlay
        raw_signal = trend_signal * (1.0 - FUNDING_WEIGHT) + funding_signal
        
        # Volatility normalization
        atr_pct = atr[i] / close[i]
        if atr_pct > 0:
            vol_factor = VOL_TARGET / atr_pct
            vol_factor = np.clip(vol_factor, 0.5, 2.0)
            raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SMOOTHING * prev_signal + (1.0 - SMOOTHING) * raw_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals