#!/usr/bin/env python3
"""
strategy.py - Mean Reversion with Trend Filter and Volatility Bands
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Mean reversion strategy with trend filter - trades against extremes
    when the longer-term trend supports the reversal direction.
    - RSI for overbought/oversold conditions
    - Bollinger Bands for price position relative to volatility
    - SMA trend filter for direction bias
    - Volume confirmation for entry validity
    
    Rationale: After -97% loss on pure trend-following (#002), 
    mean reversion may work better in ranging crypto markets.

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

name = "mean_reversion_trend_filter"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for mean reversion

# Strategy parameters
RSI_PERIOD = 14               # RSI calculation period
RSI_OVERBOUGHT = 70           # RSI overbought threshold
RSI_OVERSOLD = 30             # RSI oversold threshold
BB_PERIOD = 20                # Bollinger Bands period
BB_STD = 2.0                  # Bollinger Bands standard deviations
SMA_TREND_PERIOD = 50         # SMA for trend direction filter
VOLUME_LOOKBACK = 20          # Lookback for volume average
VOLUME_THRESHOLD = 1.2        # Volume confirmation threshold
ATR_PERIOD = 14               # ATR for volatility adjustment
MIN_SIGNAL = 0.2              # Minimum signal magnitude to trade
TREND_FILTER_STRENGTH = 0.5   # How much to weight trend filter (0-1)


# =============================================================================
# Signal Generation
# =============================================================================

def calculate_sma(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average using only past data.
    
    Args:
        close: Array of close prices
        period: SMA period
    
    Returns:
        Array of SMA values
    """
    n = len(close)
    sma = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return sma
    
    # Calculate SMA using pandas for efficiency
    close_series = pd.Series(close)
    sma_values = close_series.rolling(window=period, min_periods=period).mean().values
    
    # Copy valid values
    sma[period-1:] = sma_values[period-1:]
    
    return sma


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
    delta[1:] = np.diff(close)
    
    # Separate gains and losses
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    
    # Calculate average gains and losses using EMA-like smoothing
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    
    # Initialize with SMA
    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])
    
    # Calculate using Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    
    # Calculate RS and RSI
    rs = np.zeros(n, dtype=np.float64)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi[mask] = 100.0 - (100.0 / (1.0 + rs[mask]))
    # Where avg_loss is 0, RSI = 100
    rsi[~mask] = 100.0
    
    return rsi


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands using only past data.
    
    Args:
        close: Array of close prices
        period: BB period
        std_dev: Standard deviation multiplier
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower
    
    close_series = pd.Series(close)
    
    # Calculate middle band (SMA)
    middle_series = close_series.rolling(window=period, min_periods=period).mean()
    middle[:] = middle_series.values
    
    # Calculate standard deviation
    std_series = close_series.rolling(window=period, min_periods=period).std()
    
    # Calculate bands
    upper[:] = middle_series.values + (std_dev * std_series.values)
    lower[:] = middle_series.values - (std_dev * std_series.values)
    
    return upper, middle, lower


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
    
    volume_series = pd.Series(volume)
    rolling_avg = volume_series.rolling(window=lookback, min_periods=lookback).mean().values
    
    # Avoid division by zero
    mask = rolling_avg > 0
    volume_ratio[mask] = volume[mask] / rolling_avg[mask]
    
    return volume_ratio


def calculate_bb_position(close: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """
    Calculate price position within Bollinger Bands.
    0 = at lower band, 0.5 = at middle, 1 = at upper band
    
    Args:
        close: Array of close prices
        upper: Array of upper band values
        lower: Array of lower band values
    
    Returns:
        Array of position values (0-1 range, can exceed outside bands)
    """
    n = len(close)
    bb_pos = np.zeros(n, dtype=np.float64)
    
    band_width = upper - lower
    
    # Avoid division by zero
    mask = band_width > 0
    bb_pos[mask] = (close[mask] - lower[mask]) / band_width[mask]
    
    return bb_pos


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Mean Reversion Strategy with Trend Filter.
    
    Signal Logic:
    1. Calculate RSI for overbought/oversold conditions
    2. Calculate Bollinger Bands for price position
    3. Calculate SMA for trend direction filter
    4. Calculate volume ratio for confirmation
    5. Generate mean reversion signals with trend bias
    
    Entry Conditions:
    - LONG: RSI < oversold AND price near lower BB AND (price > SMA or weak trend filter)
    - SHORT: RSI > overbought AND price near upper BB AND (price < SMA or weak trend filter)
    
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
    
    # Calculate technical indicators
    rsi = calculate_rsi(close, RSI_PERIOD)
    sma_trend = calculate_sma(close, SMA_TREND_PERIOD)
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    bb_position = calculate_bb_position(close, bb_upper, bb_lower)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Calculate RSI extremeness (how far from neutral 50)
    rsi_extreme = np.zeros(n, dtype=np.float64)
    rsi_extreme = np.where(rsi < 50, (50 - rsi) / 50.0, (rsi - 50) / 50.0)
    
    # Calculate trend direction
    trend_direction = np.zeros(n, dtype=np.float64)
    trend_mask = sma_trend > 0
    trend_direction[trend_mask] = np.where(close[trend_mask] > sma_trend[trend_mask], 1.0, -1.0)
    
    # Determine minimum valid index
    min_valid_index = max(
        SMA_TREND_PERIOD,
        BB_PERIOD,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK
    )
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip if any required data is invalid
        if close[i] <= 0 or atr[i] <= 0 or bb_upper[i] <= 0:
            signals[i] = 0.0
            continue
        
        # RSI conditions
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Bollinger Band position conditions
        bb_low = bb_position[i] < 0.2  # Price in lower 20% of bands
        bb_high = bb_position[i] > 0.8  # Price in upper 20% of bands
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_THRESHOLD
        
        # Trend filter
        trend_bullish = trend_direction[i] > 0
        trend_bearish = trend_direction[i] < 0
        
        # Calculate base signal from mean reversion logic
        raw_signal = 0.0
        
        # LONG signal: oversold RSI + lower BB + volume
        if rsi_oversold and bb_low:
            long_strength = (RSI_OVERSOLD - rsi[i]) / RSI_OVERSOLD  # 0-1 scale
            bb_strength = (0.2 - bb_position[i]) / 0.2  # 0-1 scale
            raw_signal = 0.5 * (long_strength + bb_strength)
            
            # Apply trend filter (favor longs in bullish trend)
            if trend_bullish:
                raw_signal *= (1.0 + TREND_FILTER_STRENGTH)
            else:
                raw_signal *= (1.0 - TREND_FILTER_STRENGTH)
        
        # SHORT signal: overbought RSI + upper BB + volume
        elif rsi_overbought and bb_high:
            short_strength = (rsi[i] - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT)  # 0-1 scale
            bb_strength = (bb_position[i] - 0.8) / 0.2  # 0-1 scale
            raw_signal = -0.5 * (short_strength + bb_strength)
            
            # Apply trend filter (favor shorts in bearish trend)
            if trend_bearish:
                raw_signal *= (1.0 + TREND_FILTER_STRENGTH)
            else:
                raw_signal *= (1.0 - TREND_FILTER_STRENGTH)
        
        # Apply volume confirmation (reduce signal if volume low)
        if not volume_confirmed:
            raw_signal *= 0.6
        
        # Volatility adjustment (reduce position in high volatility)
        atr_pct = atr[i] / close[i]
        vol_factor = 1.0
        if atr_pct > 0:
            # Typical 1h ATR% is 0.5-2%, scale inversely
            vol_factor = min(1.0, 0.02 / max(atr_pct, 0.001))
        
        signal = raw_signal * vol_factor
        
        # Apply minimum signal threshold
        if abs(signal) < MIN_SIGNAL:
            signal = 0.0
        
        # Clip to [-1, 1]
        signal = np.clip(signal, -1.0, 1.0)
        
        signals[i] = signal
    
    return signals