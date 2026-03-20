#!/usr/bin/env python3
"""
strategy.py - Volatility Adaptive Mean Reversion V14
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "4h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Mean reversion with trend filter on 4h timeframe:
    - Primary signal: RSI extremes (oversold/overbought) for mean reversion
    - Trend filter: Only trade mean reversion in direction of 200 EMA trend
    - Volatility scaling: Reduce position size during high volatility
    - Funding overlay: Extreme funding reinforces mean reversion signals
    - Signal decay: Gradually reduce signal magnitude to encourage exits
    
    Why 4h timeframe:
    - Cleaner signals than 1h (less noise)
    - More trades than 1d (better statistical significance)
    - Crypto mean reversion works well on 4h cycles
    - Lower transaction cost impact vs 5m/15m
    
    Why mean reversion:
    - Crypto is mean-reverting on shorter timeframes
    - Trend filter prevents fighting strong trends
    - Better risk/reward than pure trend following (see exp history)
    - Lower drawdown potential than trend strategies

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

name = "vol_adaptive_meanrev_v14"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for drawdown control

# EMA configuration for trend filter
EMA_TREND = 50
EMA_MAJOR = 200

# RSI configuration for mean reversion
RSI_PERIOD = 14
RSI_OVERSOLD = 30  # Entry long when RSI below this
RSI_OVERBOUGHT = 70  # Entry short when RSI above this
RSI_EXIT_LONG = 50  # Exit long when RSI above this
RSI_EXIT_SHORT = 50  # Exit short when RSI below this

# Funding rate configuration
FUNDING_EXTREME_THRESHOLD = 0.0008  # 0.08% per 8hr
FUNDING_WEIGHT = 0.30  # How much funding affects signal

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.020  # Target ATR as % of price
VOLATILITY_MIN = 0.005  # Minimum ATR % to trade
VOLATILITY_MAX = 0.080  # Maximum ATR % to trade
VOLATILITY_PENALTY = 0.5  # Reduce signal by this factor when vol is high

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.70  # Maximum signal magnitude (conservative)
SIGNAL_DECAY = 0.95  # Decay factor per bar (encourages exits)
TREND_FILTER_STRENGTH = 0.60  # How much trend filter affects signal

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.50  # Volume must be at least this % of average


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


def calculate_funding_signal(funding_rate: np.ndarray,
                             threshold: float = 0.0008,
                             weight: float = 0.30) -> np.ndarray:
    """
    Calculate funding rate contrarian signal.
    Extreme positive funding → short bias (negative signal)
    Extreme negative funding → long bias (positive signal)
    Returns value in [-weight, weight].
    Only uses current funding rate (no look-ahead).
    """
    n = len(funding_rate)
    signal = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        fr = funding_rate[i]
        
        if fr > threshold:
            # Extreme positive funding → short bias
            signal[i] = -weight * min(1.0, fr / threshold)
        elif fr < -threshold:
            # Extreme negative funding → long bias
            signal[i] = weight * min(1.0, abs(fr) / threshold)
        else:
            # Scale linearly in moderate range
            signal[i] = -weight * (fr / threshold)
    
    return np.clip(signal, -weight, weight)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Volatility Adaptive Mean Reversion V14 Strategy.
    
    Signal Logic:
    1. Calculate RSI for mean reversion entry/exit signals
    2. Calculate trend filter (price vs 200 EMA)
    3. Calculate funding rate overlay
    4. Combine: RSI signal * trend filter + funding overlay
    5. Apply volatility-based position sizing
    6. Apply signal decay to encourage exits
    7. Filter by minimum signal magnitude
    
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
    ema_trend = calculate_ema(close, EMA_TREND)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    funding_signal = calculate_funding_signal(
        funding_rate, FUNDING_EXTREME_THRESHOLD, FUNDING_WEIGHT
    )
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_TREND,
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
        
        # === RSI Mean Reversion Signal ===
        rsi_signal = 0.0
        
        if rsi[i] < RSI_OVERSOLD:
            # Oversold → long bias
            rsi_signal = 1.0 * ((RSI_OVERSOLD - rsi[i]) / RSI_OVERSOLD)
        elif rsi[i] > RSI_OVERBOUGHT:
            # Overbought → short bias
            rsi_signal = -1.0 * ((rsi[i] - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT))
        elif prev_signal > 0 and rsi[i] > RSI_EXIT_LONG:
            # Exit long position
            rsi_signal = 0.0
        elif prev_signal < 0 and rsi[i] < RSI_EXIT_SHORT:
            # Exit short position
            rsi_signal = 0.0
        else:
            # Hold current position or neutral
            rsi_signal = prev_signal * SIGNAL_DECAY
        
        # === Trend Filter ===
        # Only trade mean reversion in direction of major trend
        trend_direction = np.sign(close[i] - ema_major[i])
        trend_strength = abs(close[i] - ema_major[i]) / close[i]
        trend_strength = np.clip(trend_strength * 100, 0.0, 1.0)
        
        # Apply trend filter to RSI signal
        if rsi_signal > 0:
            # Long signal: only if trend is up or neutral
            if trend_direction < 0:
                rsi_signal *= (1.0 - TREND_FILTER_STRENGTH)
        elif rsi_signal < 0:
            # Short signal: only if trend is down or neutral
            if trend_direction > 0:
                rsi_signal *= (1.0 - TREND_FILTER_STRENGTH)
        
        # === Combine with Funding Signal ===
        raw_signal = rsi_signal * 0.70 + funding_signal[i] * 0.30
        
        # === Volatility-Based Position Sizing ===
        # Reduce position size when volatility is high
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, VOLATILITY_PENALTY, 1.5)
        raw_signal *= vol_factor
        
        # === Signal Smoothing and Decay ===
        # Apply decay to previous signal to encourage exits
        if abs(raw_signal) < abs(prev_signal) * SIGNAL_DECAY:
            smoothed_signal = prev_signal * SIGNAL_DECAY
        else:
            smoothed_signal = raw_signal
        
        # === Apply Minimum Magnitude Filter ===
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # === Clip to Max Signal ===
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals