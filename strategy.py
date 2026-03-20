#!/usr/bin/env python3
"""
strategy.py - Mean Reversion with Trend Filter and Volume Confirmation
=======================================================================
Strategy Hypothesis:
    Crypto markets spend significant time in ranges. This strategy:
    1. Uses Bollinger Bands to identify overextended prices
    2. Filters entries with longer-term trend direction
    3. Confirms with volume spikes for reversal validity
    4. Reduces position size during high volatility regimes

Look-Ahead Safety:
    - All rolling calculations use only past data
    - No .shift(-n) or future index access
    - Signal at bar t uses only prices.iloc[:t+1]
"""

import numpy as np
import pandas as pd

# =============================================================================
# Strategy Configuration
# =============================================================================

name = "mean_reversion_trend_filter_v2"
timeframe = "1h"
leverage = 2.0  # Conservative for mean reversion

# Bollinger Band parameters
BB_PERIOD = 20
BB_STD = 2.0

# Trend filter (longer timeframe bias)
TREND_EMA = 100

# RSI for overbought/oversold confirmation
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_THRESHOLD = 1.3

# Volatility filter
ATR_PERIOD = 14
VOLATILITY_CAP = 0.03  # Max ATR% to trade

# Signal thresholds
MIN_SIGNAL = 0.25


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_sma(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average using only past data."""
    n = len(close)
    sma = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return sma
    
    # Use pandas for efficient rolling calculation
    sma_series = pd.Series(close).rolling(window=period, min_periods=period).mean()
    sma[:] = sma_series.values
    
    return sma


def calculate_std(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate rolling standard deviation using only past data."""
    n = len(close)
    std = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return std
    
    std_series = pd.Series(close).rolling(window=period, min_periods=period).std()
    std[:] = std_series.values
    
    return std


def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average using only past data."""
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return ema
    
    ema[period - 1] = np.mean(close[:period])
    multiplier = 2.0 / (period + 1)
    
    for i in range(period, n):
        ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI using only past data."""
    n = len(close)
    rsi = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return rsi
    
    delta = np.zeros(n, dtype=np.float64)
    delta[1:] = np.diff(close)
    
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    
    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi[period:] = 100 - (100 / (1 + rs[period:]))
    
    return rsi


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average True Range using only past data."""
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
    
    atr[period - 1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Calculate volume ratio relative to rolling average."""
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    rolling_avg = volume_series.rolling(window=lookback, min_periods=lookback).mean().values
    
    mask = rolling_avg > 0
    volume_ratio[mask] = volume[mask] / rolling_avg[mask]
    
    return volume_ratio


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Mean Reversion Strategy with Trend Filter.
    
    Logic:
    1. Price outside Bollinger Bands suggests overextension
    2. RSI confirms overbought/oversold condition
    3. Volume spike confirms reversal potential
    4. Trend EMA filters against strong trending moves
    5. ATR filter avoids high volatility regimes
    
    Entry:
    - LONG: Price < Lower BB AND RSI < 30 AND volume confirmed
    - SHORT: Price > Upper BB AND RSI > 70 AND volume confirmed
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume]
    
    Returns:
        np.ndarray of signals in [-1, 1]
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract columns with error handling
    try:
        close = prices["close"].values.astype(np.float64)
        high = prices["high"].values.astype(np.float64)
        low = prices["low"].values.astype(np.float64)
        volume = prices["volume"].values.astype(np.float64)
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Handle NaN and invalid values
    close = np.nan_to_num(close, nan=0.0, posinf=1.0, neginf=1.0)
    high = np.nan_to_num(high, nan=0.0, posinf=1.0, neginf=1.0)
    low = np.nan_to_num(low, nan=0.0, posinf=1.0, neginf=1.0)
    volume = np.nan_to_num(volume, nan=0.0, posinf=1.0, neginf=1.0)
    
    # Ensure positive prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate indicators
    bb_mid = calculate_sma(close, BB_PERIOD)
    bb_std = calculate_std(close, BB_PERIOD)
    bb_upper = bb_mid + BB_STD * bb_std
    bb_lower = bb_mid - BB_STD * bb_std
    
    trend_ema = calculate_ema(close, TREND_EMA)
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Determine minimum valid index
    min_valid = max(
        BB_PERIOD,
        TREND_EMA,
        RSI_PERIOD + 1,
        ATR_PERIOD,
        VOLUME_LOOKBACK
    )
    
    # Generate signals
    for i in range(min_valid, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0 or bb_std[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Volatility filter - skip if too volatile
        atr_pct = atr[i] / close[i]
        if atr_pct > VOLATILITY_CAP:
            signals[i] = 0.0
            continue
        
        # Price position relative to Bollinger Bands
        price_position = (close[i] - bb_mid[i]) / bb_std[i]  # Normalized distance
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_THRESHOLD
        
        # RSI signal strength
        rsi_signal = 0.0
        if rsi[i] < RSI_OVERSOLD:
            rsi_signal = (RSI_OVERSOLD - rsi[i]) / RSI_OVERSOLD  # 0 to 1
        elif rsi[i] > RSI_OVERBOUGHT:
            rsi_signal = -(rsi[i] - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT)  # 0 to -1
        
        # BB signal strength
        bb_signal = 0.0
        if close[i] < bb_lower[i]:
            bb_signal = (bb_lower[i] - close[i]) / bb_std[i]  # Positive for long
        elif close[i] > bb_upper[i]:
            bb_signal = -(close[i] - bb_upper[i]) / bb_std[i]  # Negative for short
        
        # Trend filter - reduce signal if trading against strong trend
        trend_factor = 1.0
        if trend_ema[i] > 0:
            trend_pct = (close[i] - trend_ema[i]) / trend_ema[i]
            # If price far above trend EMA, reduce short signals
            if trend_pct > 0.05 and bb_signal < 0:
                trend_factor = 0.5
            # If price far below trend EMA, reduce long signals
            elif trend_pct < -0.05 and bb_signal > 0:
                trend_factor = 0.5
        
        # Combine signals
        raw_signal = 0.6 * bb_signal + 0.4 * rsi_signal
        
        # Apply volume confirmation
        if not volume_confirmed:
            raw_signal *= 0.6
        
        # Apply trend filter
        raw_signal *= trend_factor
        
        # Scale signal based on overextension magnitude
        signal = np.clip(raw_signal, -1.0, 1.0)
        
        # Apply minimum threshold
        if abs(signal) < MIN_SIGNAL:
            signal = 0.0
        
        signals[i] = signal
    
    return signals