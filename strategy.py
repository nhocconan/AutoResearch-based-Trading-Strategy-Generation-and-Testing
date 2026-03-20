#!/usr/bin/env python3
"""
strategy.py - Mean Reversion with Trend Filter and RSI Divergence
=================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Mean-reversion strategy with trend filter - opposite of pure trend-following.
    - Use 200 EMA for long-term trend bias (not entry trigger)
    - RSI(14) for mean-reversion entries (oversold in uptrend = long)
    - RSI divergence detection for stronger signals
    - Volume confirmation for reversal validity
    - ATR-based volatility scaling for position sizing
    - Conservative leverage to account for crypto volatility

    Rationale: Crypto markets often mean-revert on 1h timeframe. Pure trend-
    following failed in experiment #001. This strategy buys dips in uptrends
    and sells rallies in downtrends.

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

name = "mean_reversion_rsi_trend_filter"
timeframe = "1h"
leverage = 2.5  # Slightly higher than previous due to mean-reversion nature

# Strategy parameters
EMA_TREND = 200               # Long-term trend filter (not entry trigger)
RSI_PERIOD = 14               # RSI calculation period
RSI_OVERBOUGHT = 65           # RSI overbought threshold (lower than typical)
RSI_OVERSOLD = 35             # RSI oversold threshold (higher than typical)
RSI_EXTREME_OVERBOUGHT = 75   # Extreme overbought for stronger signals
RSI_EXTREME_OVERSOLD = 25     # Extreme oversold for stronger signals
VOLUME_LOOKBACK = 20          # Lookback for volume average
VOLUME_THRESHOLD = 1.2        # Volume spike multiplier
ATR_PERIOD = 14               # ATR calculation period
VOLATILITY_TARGET = 0.015     # Target volatility for position sizing
MIN_SIGNAL = 0.15             # Minimum signal magnitude to trade
MAX_SIGNAL = 0.85             # Maximum signal magnitude
DIVERGENCE_LOOKBACK = 5       # Lookback for RSI divergence detection


# =============================================================================
# Helper Functions
# =============================================================================

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
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        if delta[i] > 0:
            gains[i] = delta[i]
        else:
            losses[i] = -delta[i]
    
    # Calculate initial average gain/loss using SMA
    avg_gain = np.mean(gains[1:period+1])
    avg_loss = np.mean(losses[1:period+1])
    
    rsi[period] = 50.0  # Default if no data
    
    if avg_loss != 0:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate RSI using Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        else:
            rsi[i] = 100.0
    
    return rsi


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


def detect_rsi_divergence(close: np.ndarray, rsi: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Detect RSI divergence using only past data.
    
    Bullish divergence: Price makes lower low, RSI makes higher low
    Bearish divergence: Price makes higher high, RSI makes lower high
    
    Args:
        close: Array of close prices
        rsi: Array of RSI values
        lookback: Lookback period for divergence detection
    
    Returns:
        Array of divergence signals: 1=bullish, -1=bearish, 0=none
    """
    n = len(close)
    divergence = np.zeros(n, dtype=np.float64)
    
    if n < lookback * 2:
        return divergence
    
    for i in range(lookback * 2, n):
        # Find local extrema in lookback window
        window_close = close[i-lookback:i+1]
        window_rsi = rsi[i-lookback:i+1]
        
        if len(window_close) < 3 or len(window_rsi) < 3:
            continue
        
        # Check for bullish divergence (price lower low, RSI higher low)
        price_min_idx = np.argmin(window_close)
        rsi_min_idx = np.argmin(window_rsi)
        
        # Check recent vs earlier in window
        if price_min_idx > lookback // 2:
            earlier_price_min = np.min(window_close[:lookback//2+1])
            earlier_rsi_min = np.min(window_rsi[:lookback//2+1])
            
            if window_close[price_min_idx] < earlier_price_min and window_rsi[price_min_idx] > earlier_rsi_min:
                divergence[i] = 1.0
                continue
        
        # Check for bearish divergence (price higher high, RSI lower high)
        price_max_idx = np.argmax(window_close)
        rsi_max_idx = np.argmax(window_rsi)
        
        if price_max_idx > lookback // 2:
            earlier_price_max = np.max(window_close[:lookback//2+1])
            earlier_rsi_max = np.max(window_rsi[:lookback//2+1])
            
            if window_close[price_max_idx] > earlier_price_max and window_rsi[price_max_idx] < earlier_rsi_max:
                divergence[i] = -1.0
    
    return divergence


def calculate_price_position(close: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate where price sits within recent range (0=low, 1=high).
    Only uses past data.
    
    Args:
        close: Array of close prices
        lookback: Lookback period for range calculation
    
    Returns:
        Array of position values (0-1)
    """
    n = len(close)
    position = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return position
    
    for i in range(lookback, n):
        window = close[i-lookback:i+1]
        high = np.max(window)
        low = np.min(window)
        
        if high > low:
            position[i] = (close[i] - low) / (high - low)
        else:
            position[i] = 0.5
    
    return position


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Mean Reversion Strategy with Trend Filter and RSI Divergence.
    
    Signal Logic:
    1. Calculate 200 EMA for long-term trend bias
    2. Calculate RSI(14) for mean-reversion entries
    3. Detect RSI divergence for stronger signals
    4. Calculate ATR for volatility-based position sizing
    5. Calculate volume ratio for confirmation
    6. Calculate price position in recent range
    
    Entry Conditions:
    - LONG: Price > 200 EMA (uptrend) AND RSI < 35 (oversold) OR bullish divergence
    - SHORT: Price < 200 EMA (downtrend) AND RSI > 65 (overbought) OR bearish divergence
    
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
    
    # Calculate EMA for trend bias
    ema_trend = calculate_ema(close, EMA_TREND)
    
    # Calculate RSI for mean-reversion signals
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Calculate ATR for volatility adjustment
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Calculate volume ratio
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Detect RSI divergence
    divergence = detect_rsi_divergence(close, rsi, DIVERGENCE_LOOKBACK)
    
    # Calculate price position in recent range
    price_position = calculate_price_position(close, lookback=20)
    
    # Determine minimum valid index (all indicators need warmup period)
    min_valid_index = max(
        EMA_TREND,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        DIVERGENCE_LOOKBACK * 2 + 1,
        20  # price position lookback
    )
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip if any required data is invalid
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Trend bias from EMA
        trend_bullish = close[i] > ema_trend[i]
        trend_bearish = close[i] < ema_trend[i]
        
        # RSI mean-reversion signals
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        rsi_extreme_oversold = rsi[i] < RSI_EXTREME_OVERSOLD
        rsi_extreme_overbought = rsi[i] > RSI_EXTREME_OVERBOUGHT
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_THRESHOLD
        
        # Calculate RSI extremity factor (stronger signal at extremes)
        rsi_factor = 1.0
        if rsi_extreme_oversold or rsi_extreme_overbought:
            rsi_factor = 1.5
        elif rsi_oversold or rsi_overbought:
            rsi_factor = 1.0
        else:
            rsi_factor = 0.5  # Weak signal in neutral RSI
        
        # Divergence boost
        divergence_boost = 0.0
        if divergence[i] != 0:
            divergence_boost = 0.5  # Add 50% to signal strength
        
        # Price position factor (better entries at range extremes)
        position_factor = 1.0
        if price_position[i] < 0.2 or price_position[i] > 0.8:
            position_factor = 1.2  # Better entries at range edges
        
        # Volatility adjustment (reduce position in high volatility)
        atr_pct = atr[i] / close[i]
        vol_factor = 1.0
        if atr_pct > 0:
            # Scale inversely to volatility, target ~1.5% hourly volatility
            vol_factor = min(1.5, VOLATILITY_TARGET / max(atr_pct, 0.001))
        
        # Calculate base signal
        raw_signal = 0.0
        signal_confidence = 0.0
        
        # LONG signal: uptrend + oversold RSI or bullish divergence
        if trend_bullish:
            if rsi_oversold or divergence[i] == 1.0:
                signal_confidence = rsi_factor + divergence_boost
                if volume_confirmed:
                    signal_confidence *= 1.2
                if price_position[i] < 0.3:
                    signal_confidence *= position_factor
                raw_signal = signal_confidence
        
        # SHORT signal: downtrend + overbought RSI or bearish divergence
        elif trend_bearish:
            if rsi_overbought or divergence[i] == -1.0:
                signal_confidence = rsi_factor + divergence_boost
                if volume_confirmed:
                    signal_confidence *= 1.2
                if price_position[i] > 0.7:
                    signal_confidence *= position_factor
                raw_signal = -signal_confidence
        
        # Apply volatility adjustment
        signal = raw_signal * vol_factor
        
        # Apply minimum signal threshold
        if abs(signal) < MIN_SIGNAL:
            signal = 0.0
        
        # Clip to [-MAX_SIGNAL, MAX_SIGNAL] to leave room for portfolio scaling
        signal = np.clip(signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
    
    return signals