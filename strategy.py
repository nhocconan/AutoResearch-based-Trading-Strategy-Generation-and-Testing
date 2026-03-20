#!/usr/bin/env python3
"""
strategy.py - Bollinger Mean Reversion with Trend Filter
=======================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Mean reversion strategy with trend filter for crypto perpetuals.
    - Use Bollinger Bands to identify overextended prices
    - RSI confirms overbought/oversold conditions
    - Longer EMA filters trade direction (only trade with trend)
    - Volume confirms reversal validity
    
    Rationale: Crypto markets often overshoot then revert. Pure trend
    following failed in experiments #002-#006. This tests mean reversion
    with trend alignment to avoid catching falling knives.

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

name = "bb_mean_reversion_trend_filter"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for mean reversion

# Bollinger Band parameters
BB_PERIOD = 20              # Rolling window for BB
BB_STD_MULT = 2.0           # Standard deviation multiplier

# RSI parameters
RSI_PERIOD = 14             # RSI calculation period
RSI_OVERBOUGHT = 70         # Overbought threshold
RSI_OVERSOLD = 30           # Oversold threshold

# Trend filter parameters
TREND_EMA_PERIOD = 50       # Longer EMA for trend direction
TREND_STRENGTH_MIN = 0.001  # Minimum trend strength to trade

# Volume confirmation
VOLUME_LOOKBACK = 20        # Lookback for volume average
VOLUME_THRESHOLD = 1.2      # Minimum volume ratio to confirm

# Signal parameters
MIN_SIGNAL_MAGNITUDE = 0.3  # Minimum signal to execute trade
MAX_SIGNAL = 1.0            # Maximum signal magnitude


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_sma(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average using only past data.
    
    Args:
        data: Array of values
        period: SMA period
    
    Returns:
        Array of SMA values
    """
    n = len(data)
    sma = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return sma
    
    # Calculate SMA for each position
    for i in range(period - 1, n):
        sma[i] = np.mean(data[i - period + 1:i + 1])
    
    return sma


def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
    
    Args:
        close: Array of close prices
        period: EMA period
    
    Returns:
        Array of EMA values
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


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_mult: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands using only past data.
    
    Args:
        close: Array of close prices
        period: Rolling window period
        std_mult: Standard deviation multiplier
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = middle[i] + std_mult * std
        lower[i] = middle[i] - std_mult * std
    
    return upper, middle, lower


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using only past data.
    
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
    for i in range(1, n):
        delta[i] = close[i] - close[i-1]
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    
    # Calculate average gains and losses using EMA-like smoothing
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    
    # Initialize with SMA
    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])
    
    # Calculate RSI for remaining periods
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    
    # Calculate RS and RSI
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio relative to rolling average.
    Only uses past volume data (no look-ahead).
    
    Args:
        volume: Array of volume values
        lookback: Rolling window for average calculation
    
    Returns:
        Array of volume ratios
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    for i in range(lookback - 1, n):
        window = volume[i - lookback + 1:i + 1]
        avg_volume = np.mean(window)
        if avg_volume > 0:
            volume_ratio[i] = volume[i] / avg_volume
    
    return volume_ratio


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using only past data.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ATR period
    
    Returns:
        Array of ATR values
    """
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return atr
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Initialize ATR with SMA of TR
    atr[period - 1] = np.mean(tr[:period])
    
    # Calculate ATR using Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Bollinger Band Mean Reversion with Trend Filter Strategy.
    
    Signal Logic:
    1. Calculate Bollinger Bands for overextension detection
    2. Calculate RSI for overbought/oversold confirmation
    3. Calculate trend EMA for direction filter
    4. Calculate volume ratio for confirmation
    5. Generate mean reversion signals aligned with trend
    
    Entry Conditions:
    - LONG: Price < Lower BB AND RSI < Oversold AND Price > Trend EMA
    - SHORT: Price > Upper BB AND RSI > Overbought AND Price < Trend EMA
    
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
        # Return zeros if required columns missing
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
    
    # Calculate Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
        close, BB_PERIOD, BB_STD_MULT
    )
    
    # Calculate RSI
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Calculate trend EMA
    trend_ema = calculate_ema(close, TREND_EMA_PERIOD)
    
    # Calculate volume ratio
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Calculate ATR for volatility adjustment
    atr = calculate_atr(high, low, close, 14)
    
    # Calculate bandwidth for regime detection
    bandwidth = np.zeros(n, dtype=np.float64)
    for i in range(BB_PERIOD - 1, n):
        if bb_middle[i] > 0:
            bandwidth[i] = (bb_upper[i] - bb_lower[i]) / bb_middle[i]
    
    # Determine minimum valid index
    min_valid_index = max(
        BB_PERIOD,
        RSI_PERIOD + 1,
        TREND_EMA_PERIOD,
        VOLUME_LOOKBACK,
        15  # ATR period
    )
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip if any required data is invalid
        if close[i] <= 0 or bb_middle[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Price position relative to Bollinger Bands
        price_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i])
        price_position = np.clip(price_position, 0.0, 1.0)
        
        # Distance from middle band (normalized)
        dist_from_middle = (close[i] - bb_middle[i]) / bb_middle[i]
        
        # Trend direction
        price_above_trend = close[i] > trend_ema[i]
        price_below_trend = close[i] < trend_ema[i]
        
        # Trend strength (distance from EMA normalized)
        trend_strength = abs(close[i] - trend_ema[i]) / close[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_THRESHOLD
        
        # Calculate base signal from mean reversion logic
        raw_signal = 0.0
        
        # LONG signal: Price at lower band, RSI oversold, above trend EMA
        if price_position < 0.2 and rsi_oversold and price_above_trend:
            # Stronger signal when deeper in oversold territory
            signal_strength = (1.0 - price_position) * (RSI_OVERSOLD - rsi[i]) / RSI_OVERSOLD
            raw_signal = signal_strength
        
        # SHORT signal: Price at upper band, RSI overbought, below trend EMA
        elif price_position > 0.8 and rsi_overbought and price_below_trend:
            # Stronger signal when deeper in overbought territory
            signal_strength = price_position * (rsi[i] - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT)
            raw_signal = -signal_strength
        
        # Skip if no clear signal
        if raw_signal == 0.0:
            signals[i] = 0.0
            continue
        
        # Apply volume confirmation (reduce signal if volume low)
        if not volume_confirmed:
            raw_signal *= 0.5
        
        # Apply trend strength filter (only trade if trend is clear)
        if trend_strength < TREND_STRENGTH_MIN:
            raw_signal *= 0.3
        
        # Volatility adjustment (reduce position in very high volatility)
        if atr[i] > 0:
            atr_pct = atr[i] / close[i]
            # Typical 1h ATR% is 0.5-2%, reduce position if > 3%
            if atr_pct > 0.03:
                raw_signal *= 0.5
        
        # Apply minimum signal threshold
        if abs(raw_signal) < MIN_SIGNAL_MAGNITUDE:
            signals[i] = 0.0
        else:
            # Clip to [-1, 1]
            signals[i] = np.clip(raw_signal, -MAX_SIGNAL, MAX_SIGNAL)
    
    return signals