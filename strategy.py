#!/usr/bin/env python3
"""
strategy.py - RSI Mean Reversion with Trend Filter
=======================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Mean reversion within trend - buy dips in uptrend, sell rallies in downtrend
    - RSI identifies overbought/oversold conditions
    - 50 EMA filters trend direction
    - Only trade RSI signals in trend direction
    - Lower leverage for safety after #002 poor performance

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

name = "rsi_mean_reversion_trend"
timeframe = "1h"
leverage = 1.5  # Conservative leverage after poor #002 performance

# Strategy parameters
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
EMA_TREND = 50
VOLATILITY_WINDOW = 20
MIN_SIGNAL = 0.3


# =============================================================================
# Signal Generation
# =============================================================================

def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate RSI using only past data.
    
    Args:
        close: Array of close prices
        period: RSI period
    
    Returns:
        Array of RSI values (0-100)
    """
    n = len(close)
    rsi = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    delta = np.zeros(n, dtype=np.float64)
    delta[1:] = np.diff(close)
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    
    # Initialize average gain/loss with SMA
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    
    rsi[period] = 100 - (100 / (1 + avg_gain / max(avg_loss, 1e-10)))
    
    # Calculate RSI using Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        rs = avg_gain / max(avg_loss, 1e-10)
        rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
    """
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return ema
    
    # Initialize with SMA
    ema[period - 1] = np.mean(close[:period])
    
    # Calculate EMA multiplier
    multiplier = 2.0 / (period + 1)
    
    # Calculate EMA for remaining periods
    for i in range(period, n):
        ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def calculate_volatility(close: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calculate rolling volatility (standard deviation of returns).
    Only uses past data.
    """
    n = len(close)
    volatility = np.zeros(n, dtype=np.float64)
    
    if n < window:
        return volatility
    
    # Calculate returns
    returns = np.zeros(n, dtype=np.float64)
    returns[1:] = np.diff(close) / close[:-1]
    
    # Calculate rolling std
    for i in range(window, n):
        volatility[i] = np.std(returns[i-window+1:i+1])
    
    return volatility


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    RSI Mean Reversion with Trend Filter Strategy.
    
    Signal Logic:
    1. Calculate RSI for overbought/oversold conditions
    2. Calculate EMA for trend direction
    3. Calculate volatility for signal scaling
    4. Generate signals: RSI oversold + uptrend = LONG, RSI overbought + downtrend = SHORT
    
    Entry Conditions:
    - LONG: RSI < 30 AND close > EMA(50)
    - SHORT: RSI > 70 AND close < EMA(50)
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract required columns with safety checks
    try:
        close = prices["close"].values.astype(np.float64)
        high = prices["high"].values.astype(np.float64)
        low = prices["low"].values.astype(np.float64)
        volume = prices["volume"].values.astype(np.float64)
    except (KeyError, TypeError, ValueError) as e:
        return signals
    
    # Handle any NaN values in price data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Ensure no zero or negative prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate RSI
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Calculate EMA for trend filter
    ema_trend = calculate_ema(close, EMA_TREND)
    
    # Calculate volatility for signal scaling
    volatility = calculate_volatility(close, VOLATILITY_WINDOW)
    
    # Determine minimum valid index
    min_valid_index = max(RSI_PERIOD + 1, EMA_TREND, VOLATILITY_WINDOW)
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip if any required data is invalid
        if close[i] <= 0 or volatility[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Trend direction
        uptrend = close[i] > ema_trend[i]
        downtrend = close[i] < ema_trend[i]
        
        # RSI conditions
        oversold = rsi[i] < RSI_OVERSOLD
        overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Calculate RSI extremity (how far from neutral 50)
        rsi_extremity = 0.0
        if oversold:
            rsi_extremity = (RSI_OVERSOLD - rsi[i]) / RSI_OVERSOLD
        elif overbought:
            rsi_extremity = (rsi[i] - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT)
        
        # Cap extremity at 1.0
        rsi_extremity = min(rsi_extremity, 1.0)
        
        # Base signal from RSI and trend
        raw_signal = 0.0
        if oversold and uptrend:
            # Buy dip in uptrend
            raw_signal = rsi_extremity
        elif overbought and downtrend:
            # Sell rally in downtrend
            raw_signal = -rsi_extremity
        
        # Volatility adjustment (reduce position in high volatility)
        # Typical 1h volatility is 0.5-2%, scale inversely
        vol_factor = 1.0
        if volatility[i] > 0:
            vol_factor = min(1.0, 0.015 / max(volatility[i], 0.001))
        
        signal = raw_signal * vol_factor
        
        # Apply minimum signal threshold
        if abs(signal) < MIN_SIGNAL:
            signal = 0.0
        
        # Clip to [-1, 1]
        signal = np.clip(signal, -1.0, 1.0)
        
        signals[i] = signal
    
    return signals