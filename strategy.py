#!/usr/bin/env python3
"""
strategy.py - Funding Mean Reversion + Trend Filter V12
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simplified approach focusing on funding rate mean reversion with trend filter:
    - Primary signal: Extreme funding rates indicate crowded positions → revert
    - Trend filter: Only take mean reversion trades against strong trends
    - RSI confirmation: Wait for momentum exhaustion before entry
    - Reduced filtering: Less aggressive volatility/regime filters
    - Cleaner signal generation: Direct mapping from indicators to signals
    
    Key changes from v11:
    - Removed excessive regime complexity
    - Lowered minimum signal thresholds
    - Reduced smoothing to allow faster signal response
    - Focus on funding rate as primary alpha source
    - Simplified trend confirmation using EMA stack

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

name = "funding_mean_reversion_v12"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for mean reversion strategy

# EMA configuration for trend filter
EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_EXTREME_HIGH = 80
RSI_EXTREME_LOW = 20

# Funding rate configuration (primary signal)
FUNDING_EXTREME_THRESHOLD = 0.0010  # 0.10% per 8hr = very extreme
FUNDING_MODERATE_THRESHOLD = 0.0005  # 0.05% per 8hr = moderate
FUNDING_LOOKBACK = 100  # For calculating extremes
FUNDING_WEIGHT = 0.60  # Weight of funding signal in final output

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_MIN_PCT = 0.001  # 0.1% minimum ATR
VOLATILITY_MAX_PCT = 0.050  # 5% maximum ATR

# Signal configuration
MAX_SIGNAL = 1.0
SMOOTHING_FACTOR = 0.40  # Lower = faster response
MIN_SIGNAL_MAGNITUDE = 0.15  # Minimum magnitude to generate non-zero signal

# Trend strength configuration
TREND_STRENGTH_THRESHOLD = 0.30  # Minimum trend strength to filter mean reversion


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


def calculate_funding_zscore(funding_rate: np.ndarray, lookback: int = 100) -> np.ndarray:
    """
    Calculate rolling z-score of funding rate for normalized extreme detection.
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    zscore = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return zscore
    
    funding_series = pd.Series(funding_rate)
    rolling_mean = funding_series.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = funding_series.rolling(window=lookback, min_periods=lookback).std()
    
    zscore = np.nan_to_num((funding_series.values - rolling_mean.values) / 
                           (rolling_std.values + 1e-10), nan=0.0)
    
    return zscore


def calculate_funding_signal(funding_rate: np.ndarray, funding_zscore: np.ndarray,
                             extreme_threshold: float = 0.0010,
                             moderate_threshold: float = 0.0005) -> np.ndarray:
    """
    Calculate funding rate mean reversion signal.
    Extreme positive funding → short bias (negative signal)
    Extreme negative funding → long bias (positive signal)
    Returns value in [-1, 1].
    Only uses current/past funding rate (no look-ahead).
    """
    n = len(funding_rate)
    signal = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        funding = funding_rate[i]
        zscore = funding_zscore[i]
        
        # Base signal from funding rate direction (mean reversion)
        if funding > extreme_threshold:
            # Very extreme positive funding → strong short signal
            signal[i] = -np.clip(funding / extreme_threshold, 0.5, 1.0)
        elif funding < -extreme_threshold:
            # Very extreme negative funding → strong long signal
            signal[i] = np.clip(-funding / extreme_threshold, 0.5, 1.0)
        elif funding > moderate_threshold:
            # Moderate positive funding → weak short signal
            signal[i] = -0.3 * (funding / moderate_threshold)
        elif funding < -moderate_threshold:
            # Moderate negative funding → weak long signal
            signal[i] = 0.3 * (-funding / moderate_threshold)
        else:
            signal[i] = 0.0
        
        # Amplify based on z-score (statistical extremeness)
        if abs(zscore) > 2.0:
            zscore_factor = np.clip(abs(zscore) / 3.0, 1.0, 1.5)
            signal[i] *= zscore_factor
    
    return signal


def calculate_trend_direction(ema_fast: float, ema_medium: float, 
                              ema_slow: float, ema_major: float) -> float:
    """
    Calculate trend direction score based on EMA stack.
    Returns value in [-1, 1] where sign is direction, magnitude is confidence.
    """
    if ema_major <= 0:
        return 0.0
    
    # Check EMA alignment
    bullish_alignment = (ema_fast > ema_medium > ema_slow > ema_major)
    bearish_alignment = (ema_fast < ema_medium < ema_slow < ema_major)
    
    # Calculate deviation from major EMA
    fast_dev = (ema_fast - ema_major) / ema_major
    medium_dev = (ema_medium - ema_major) / ema_major
    slow_dev = (ema_slow - ema_major) / ema_major
    
    avg_dev = (fast_dev + medium_dev + slow_dev) / 3
    
    if bullish_alignment:
        trend_score = np.clip(avg_dev * 10, 0.1, 1.0)
    elif bearish_alignment:
        trend_score = -np.clip(abs(avg_dev) * 10, 0.1, 1.0)
    else:
        # Partial alignment
        if avg_dev > 0:
            trend_score = np.clip(avg_dev * 5, 0.05, 0.5)
        elif avg_dev < 0:
            trend_score = -np.clip(abs(avg_dev) * 5, 0.05, 0.5)
        else:
            trend_score = 0.0
    
    return trend_score


def calculate_rsi_signal(rsi: float, overbought: float = 70, oversold: float = 30,
                         extreme_high: float = 80, extreme_low: float = 20) -> float:
    """
    Calculate RSI mean reversion signal.
    High RSI → short bias, Low RSI → long bias
    Returns value in [-1, 1].
    """
    if rsi > extreme_high:
        # Extreme overbought → strong short
        return -np.clip((rsi - extreme_high) / (100 - extreme_high), 0.5, 1.0)
    elif rsi < extreme_low:
        # Extreme oversold → strong long
        return np.clip((extreme_low - rsi) / extreme_low, 0.5, 1.0)
    elif rsi > overbought:
        # Overbought → weak short
        return -0.3 * ((rsi - overbought) / (extreme_high - overbought))
    elif rsi < oversold:
        # Oversold → weak long
        return 0.3 * ((oversold - rsi) / (oversold - extreme_low))
    else:
        return 0.0


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Funding Mean Reversion + Trend Filter V12 Strategy.
    
    Signal Logic:
    1. Funding rate extremes → primary mean reversion signal
    2. EMA stack → trend direction filter (reduce trades against strong trends)
    3. RSI → entry timing confirmation
    4. Combined signal with funding as primary driver
    5. Light smoothing for signal stability
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, funding_rate, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
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
    
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_medium = calculate_ema(close, EMA_MEDIUM)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    funding_zscore = calculate_funding_zscore(funding_rate, FUNDING_LOOKBACK)
    funding_signal = calculate_funding_signal(funding_rate, funding_zscore,
                                               FUNDING_EXTREME_THRESHOLD,
                                               FUNDING_MODERATE_THRESHOLD)
    
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        FUNDING_LOOKBACK
    )
    
    prev_signal = 0.0
    
    for i in range(min_valid_index, n):
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN_PCT or atr_pct > VOLATILITY_MAX_PCT:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        trend_direction = calculate_trend_direction(
            ema_fast[i], ema_medium[i], ema_slow[i], ema_major[i]
        )
        
        rsi_sig = calculate_rsi_signal(rsi[i], RSI_OVERBOUGHT, RSI_OVERSOLD,
                                        RSI_EXTREME_HIGH, RSI_EXTREME_LOW)
        
        fund_sig = funding_signal[i]
        
        trend_filter = 1.0
        if abs(trend_direction) > TREND_STRENGTH_THRESHOLD:
            if fund_sig > 0 and trend_direction > 0:
                trend_filter = 0.5
            elif fund_sig < 0 and trend_direction < 0:
                trend_filter = 0.5
        
        combined_signal = FUNDING_WEIGHT * fund_sig + (1.0 - FUNDING_WEIGHT) * rsi_sig
        combined_signal *= trend_filter
        
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * combined_signal
        
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals