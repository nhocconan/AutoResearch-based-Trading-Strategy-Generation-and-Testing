#!/usr/bin/env python3
"""
strategy.py - Trend Volatility Hybrid V14
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    4h trend-following with volatility and funding filters:
    - Primary signal: 50/200 EMA crossover for major trend direction
    - Entry timing: 21 EMA pullback in direction of major trend
    - Filter: ATR volatility filter (skip extreme volatility periods)
    - Confirmation: RSI momentum (not overbought/oversold at entry)
    - Overlay: Funding rate contrarian signal (avoid crowded trades)
    
    Why 4h timeframe:
    - Cleaner trends than 1h/15m, fewer false breakouts
    - Lower transaction costs relative to signal frequency
    - Better suited for trend-following strategies
    - Reduced whipsaw compared to lower timeframes
    
    Why this should control drawdown:
    - Major trend filter (200 EMA) prevents counter-trend trades
    - Volatility filter skips chaotic market periods
    - Funding overlay avoids entering at crowded extremes
    - Conservative leverage (1.5x) reduces position risk

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

name = "trend_volatility_hybrid_v14"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for drawdown control

# EMA configuration for trend detection
EMA_FAST = 21      # Entry trigger EMA
EMA_MEDIUM = 50    # Intermediate trend
EMA_MAJOR = 200    # Major trend filter

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_MIN = 40   # Minimum RSI for long entry
RSI_LONG_MAX = 65   # Maximum RSI for long entry (avoid overbought)
RSI_SHORT_MIN = 35  # Minimum RSI for short entry (avoid oversold)
RSI_SHORT_MAX = 60  # Maximum RSI for short entry

# Funding rate configuration
FUNDING_EXTREME_THRESHOLD = 0.0008  # 0.08% per 8hr = extreme
FUNDING_LOOKBACK = 50  # For calculating extremes (shorter for 4h)
FUNDING_WEIGHT = 0.30  # How much funding affects signal

# Volatility configuration
ATR_PERIOD = 14
ATR_VOLATILITY_MIN = 0.005  # Minimum ATR % to trade (avoid dead markets)
ATR_VOLATILITY_MAX = 0.040  # Maximum ATR % to trade (avoid chaos)
ATR_SMOOTHING = 7  # Smooth ATR for signal scaling

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.40  # EMA smoothing for signals
TREND_CONFIRMATION_BARS = 2  # Bars of trend confirmation needed

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


def calculate_funding_percentile(funding_rate: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calculate rolling percentile of funding rate.
    Returns value in [0, 1] representing where current funding sits in recent history.
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    percentile = np.full(n, 0.5, dtype=np.float64)
    
    if n < lookback:
        return percentile
    
    funding_series = pd.Series(funding_rate)
    
    for i in range(lookback - 1, n):
        window = funding_series.iloc[i - lookback + 1:i + 1]
        if len(window) >= lookback:
            percentile[i] = (window <= funding_rate[i]).sum() / len(window)
    
    return percentile


def calculate_funding_signal(funding_percentile: np.ndarray, 
                             funding_rate: np.ndarray,
                             extreme_threshold: float = 0.0008,
                             weight: float = 0.30) -> np.ndarray:
    """
    Calculate funding rate contrarian signal.
    High percentile (crowded longs) → short bias (negative signal)
    Low percentile (crowded shorts) → long bias (positive signal)
    Returns value in [-weight, weight].
    Only uses current/past funding rate (no look-ahead).
    """
    n = len(funding_percentile)
    signal = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        pct = funding_percentile[i]
        fr = funding_rate[i]
        
        # Contrarian signal based on percentile
        # High percentile (>0.8) = crowded longs = short bias
        # Low percentile (<0.2) = crowded shorts = long bias
        if pct > 0.80:
            # Very crowded longs
            signal[i] = -weight * ((pct - 0.80) / 0.20)
        elif pct < 0.20:
            # Very crowded shorts
            signal[i] = weight * ((0.20 - pct) / 0.20)
        elif pct > 0.65:
            # Moderately crowded longs
            signal[i] = -weight * 0.3 * ((pct - 0.65) / 0.15)
        elif pct < 0.35:
            # Moderately crowded shorts
            signal[i] = weight * 0.3 * ((0.35 - pct) / 0.15)
        else:
            signal[i] = 0.0
        
        # Also consider absolute funding rate
        if abs(fr) > extreme_threshold:
            # Extreme funding reinforces contrarian signal
            if fr > 0:
                signal[i] = max(signal[i], -weight * min(1.0, fr / extreme_threshold))
            else:
                signal[i] = min(signal[i], weight * min(1.0, abs(fr) / extreme_threshold))
    
    return signal


def calculate_trend_direction(close: np.ndarray,
                              ema_fast: np.ndarray,
                              ema_medium: np.ndarray,
                              ema_major: np.ndarray) -> np.ndarray:
    """
    Calculate trend direction based on EMA alignment.
    Returns: 1 (bullish), -1 (bearish), 0 (neutral/unclear)
    Only uses current/past data (no look-ahead).
    """
    n = len(close)
    trend = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if ema_major[i] <= 0:
            trend[i] = 0.0
            continue
        
        # Major trend: price vs 200 EMA
        major_bullish = close[i] > ema_major[i]
        major_bearish = close[i] < ema_major[i]
        
        # Medium trend: 50 EMA vs 200 EMA
        medium_bullish = ema_medium[i] > ema_major[i]
        medium_bearish = ema_medium[i] < ema_major[i]
        
        # Fast trend: 21 EMA vs 50 EMA
        fast_bullish = ema_fast[i] > ema_medium[i]
        fast_bearish = ema_fast[i] < ema_medium[i]
        
        # Count bullish/bearish signals
        bullish_count = sum([major_bullish, medium_bullish, fast_bullish])
        bearish_count = sum([major_bearish, medium_bearish, fast_bearish])
        
        # Require at least 2/3 alignment for clear trend
        if bullish_count >= 2:
            trend[i] = 1.0
        elif bearish_count >= 2:
            trend[i] = -1.0
        else:
            trend[i] = 0.0
    
    return trend


def check_rsi_entry(rsi: float, direction: int) -> bool:
    """
    Check if RSI confirms entry in given direction.
    Only uses current RSI value (no look-ahead).
    """
    if direction > 0:  # Long
        return RSI_LONG_MIN <= rsi <= RSI_LONG_MAX
    elif direction < 0:  # Short
        return RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX
    else:
        return False


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Volatility Hybrid V14 Strategy.
    
    Signal Logic:
    1. Calculate major trend direction using EMA alignment (21/50/200)
    2. Calculate funding contrarian signal from percentile extremes
    3. Check RSI for entry timing confirmation
    4. Apply volatility filter (ATR-based)
    5. Apply volume filter
    6. Combine signals with trend as primary driver
    7. Smooth signals and apply hysteresis
    8. Filter by minimum signal magnitude
    
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
    ema_medium = calculate_ema(close, EMA_MEDIUM)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    funding_percentile = calculate_funding_percentile(funding_rate, FUNDING_LOOKBACK)
    funding_signal = calculate_funding_signal(
        funding_percentile, funding_rate,
        FUNDING_EXTREME_THRESHOLD, FUNDING_WEIGHT
    )
    
    # Calculate trend direction
    trend_direction = calculate_trend_direction(
        close, ema_fast, ema_medium, ema_major
    )
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_MEDIUM,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        FUNDING_LOOKBACK,
        TREND_CONFIRMATION_BARS
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    trend_confirmed_bars = 0
    last_trend_direction = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            trend_confirmed_bars = 0
            continue
        
        # Volatility filter (not too low, not too high)
        atr_pct = atr[i] / close[i]
        if atr_pct < ATR_VOLATILITY_MIN or atr_pct > ATR_VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            trend_confirmed_bars = 0
            continue
        
        # Volume filter (ensure sufficient liquidity)
        if volume_ratio[i] < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            trend_confirmed_bars = 0
            continue
        
        # Get current trend direction
        current_trend = trend_direction[i]
        
        # Require trend confirmation (multiple bars in same direction)
        if current_trend != 0 and current_trend == last_trend_direction:
            trend_confirmed_bars += 1
        elif current_trend != 0:
            trend_confirmed_bars = 1
            last_trend_direction = current_trend
        else:
            trend_confirmed_bars = 0
            last_trend_direction = 0
        
        # Need confirmed trend before trading
        if trend_confirmed_bars < TREND_CONFIRMATION_BARS:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Check RSI entry confirmation
        if not check_rsi_entry(rsi[i], current_trend):
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Calculate base signal from trend
        base_signal = current_trend * 0.7
        
        # Get funding overlay
        fund_sig = funding_signal[i]
        
        # Combine signals: trend is primary, funding adjusts
        # If funding conflicts strongly with trend, reduce signal
        if np.sign(base_signal) != np.sign(fund_sig) and abs(fund_sig) > 0.15:
            # Strong conflict - reduce position
            raw_signal = base_signal * 0.5 + fund_sig * 0.5
        else:
            # Aligned or weak funding signal
            raw_signal = base_signal * 0.75 + fund_sig * 0.25
        
        # Volatility scaling (reduce position in high volatility)
        vol_factor = 1.0
        if atr_pct > ATR_VOLATILITY_MAX * 0.7:
            vol_factor = 0.7
        elif atr_pct < ATR_VOLATILITY_MIN * 1.5:
            vol_factor = 0.8
        
        raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # Hysteresis: don't flip direction on small changes
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < 0.15:
                smoothed_signal = prev_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals