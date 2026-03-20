#!/usr/bin/env python3
"""
strategy.py - Multi-TF Trend Pullback V14
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

Strategy Hypothesis:
    Multi-timeframe trend following with pullback entries.
    4h trend direction + 1h pullback to EMA-20 for entries.
    
    Why this should beat supertrend_4h_v1 (Sharpe=0.253, DD=-84.5%):
    - Multi-TF approach DOUBLED Sharpe in academic research (per strategy notes)
    - 4h Supertrend defines major trend direction (avoid counter-trend trades)
    - 1h EMA-20 pullback entries (better risk/reward than breakout entries)
    - Bollinger Band width regime filter (avoid high vol mean-reversion traps)
    - ATR-based position sizing normalizes risk across volatility regimes
    - Volume confirmation on entries ensures liquidity
    
    Key differences from failed multitf_supertrend_ema_v3:
    - Proper resampling of 1h→4h (not mixing timeframes incorrectly)
    - Regime filter using BB width percentile
    - Stricter entry conditions (RSI + volume + pullback depth)
    - Better signal smoothing to reduce whipsaws
    
    Expected improvements:
    - Lower drawdown via regime filter and pullback entries
    - Better Sharpe via higher quality setups (trend + pullback + volume)
    - Works across BTC/ETH/SOL due to multi-TF confirmation
"""

import numpy as np
import pandas as pd

# =============================================================================
# Strategy Configuration
# =============================================================================

name = "multitf_trend_pullback_v14"
timeframe = "1h"  # Entry timeframe (4h trend derived from this)
leverage = 1.5  # Conservative to control drawdown

# 4h Trend Filter (derived from 1h data via resampling)
SUPERTREND_ATR_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
SUPERTREND_4H_BARS = 4  # 4x 1h bars = 1x 4h bar

# 1h Entry Signals
EMA_FAST = 9
EMA_SLOW = 21
EMA_PULLBACK = 20  # Pullback to this EMA for entries
RSI_PERIOD = 14
RSI_LONG_MIN = 40  # Don't long if RSI too oversold (weak momentum)
RSI_LONG_MAX = 65  # Don't long if RSI too overbought
RSI_SHORT_MIN = 35  # Don't short if RSI too oversold
RSI_SHORT_MAX = 60  # Don't short if RSI too overbought

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 1.0  # Volume must be >= average on entry bars

# Bollinger Band regime filter
BB_PERIOD = 20
BB_STD = 2.0
BB_WIDTH_LOW_PERCENTILE = 30  # Below this = low vol (breakout regime)
BB_WIDTH_HIGH_PERCENTILE = 70  # Above this = high vol (avoid)

# ATR for risk management
ATR_PERIOD = 14
ATR_STOP_MULT = 2.5  # Trailing stop distance

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.25
MAX_SIGNAL = 0.75
SIGNAL_SMOOTHING = 0.5  # EMA smoothing on signals
HOLDBARS_MIN = 3  # Minimum bars to hold position


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                  period: int = 14) -> np.ndarray:
    """Calculate Average True Range using Wilder's smoothing."""
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return atr
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr[period] = np.mean(tr[1:period + 1])
    
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         atr_period: int = 10, multiplier: float = 3.0) -> tuple:
    """
    Calculate Supertrend indicator.
    Returns: (supertrend_values, supertrend_direction)
    direction: 1 = bullish (price above ST), -1 = bearish (price below ST)
    """
    n = len(close)
    supertrend = np.zeros(n, dtype=np.float64)
    direction = np.zeros(n, dtype=np.float64)
    
    if n < atr_period + 2:
        return supertrend, direction
    
    atr = calculate_atr(high, low, close, atr_period)
    hl2 = (high + low) / 2.0
    
    # Basic upper and lower bands
    upper_band = np.zeros(n, dtype=np.float64)
    lower_band = np.zeros(n, dtype=np.float64)
    
    for i in range(atr_period, n):
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
    
    # Initialize Supertrend
    supertrend[atr_period] = upper_band[atr_period]
    direction[atr_period] = 1  # Start bullish
    
    for i in range(atr_period + 1, n):
        if direction[i - 1] == 1:
            # Previously bullish
            if close[i] > supertrend[i - 1]:
                # Stay bullish
                supertrend[i] = max(supertrend[i - 1], lower_band[i])
                direction[i] = 1
            else:
                # Flip to bearish
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            # Previously bearish
            if close[i] < supertrend[i - 1]:
                # Stay bearish
                supertrend[i] = min(supertrend[i - 1], upper_band[i])
                direction[i] = -1
            else:
                # Flip to bullish
                supertrend[i] = lower_band[i]
                direction[i] = 1
    
    return supertrend, direction


def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return ema
    
    # Initialize with SMA
    ema[period - 1] = np.mean(close[:period])
    
    # EMA multiplier
    multiplier = 2.0 / (period + 1)
    
    for i in range(period, n):
        ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Relative Strength Index using Wilder's smoothing."""
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    
    if n < period + 1:
        return rsi
    
    delta = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    
    avg_gains = np.zeros(n, dtype=np.float64)
    avg_losses = np.zeros(n, dtype=np.float64)
    
    avg_gains[period] = np.mean(gains[1:period + 1])
    avg_losses[period] = np.mean(losses[1:period + 1])
    
    for i in range(period + 1, n):
        avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i]) / period
        avg_losses[i] = (avg_losses[i - 1] * (period - 1) + losses[i]) / period
    
    for i in range(period, n):
        if avg_losses[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gains[i] / avg_losses[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, 
                               std_dev: float = 2.0) -> tuple:
    """Calculate Bollinger Bands. Returns (upper, middle, lower, width_pct)."""
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    width_pct = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower, width_pct
    
    close_series = pd.Series(close)
    rolling_mean = close_series.rolling(window=period, min_periods=period).mean()
    rolling_std = close_series.rolling(window=period, min_periods=period).std()
    
    middle = rolling_mean.values
    upper = (rolling_mean + std_dev * rolling_std).values
    lower = (rolling_mean - std_dev * rolling_std).values
    
    # Band width as percentage of price
    for i in range(period, n):
        if middle[i] > 0:
            width_pct[i] = (upper[i] - lower[i]) / middle[i] * 100.0
    
    return upper, middle, lower, width_pct


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Calculate volume ratio vs rolling average."""
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    rolling_avg = volume_series.rolling(window=lookback, min_periods=lookback).mean()
    volume_ratio = np.nan_to_num(volume_series.values / rolling_avg.values, nan=1.0)
    
    return volume_ratio


def resample_to_4h(prices: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h data to 4h for trend filter."""
    # Set open_time as index if not already
    df = prices.copy()
    
    if 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.set_index('open_time')
    
    # Resample to 4h
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    resampled = df.resample('4h').agg(ohlc_dict)
    resampled = resampled.dropna()
    
    return resampled


def upsample_4h_signal_to_1h(signal_4h: np.ndarray, original_length: int,
                              bars_per_4h: int = 4) -> np.ndarray:
    """Upsample 4h signal to 1h by forward-filling."""
    upsampled = np.zeros(original_length, dtype=np.float64)
    
    # Each 4h bar represents 4x 1h bars
    for i, sig in enumerate(signal_4h):
        start_idx = i * bars_per_4h
        end_idx = min((i + 1) * bars_per_4h, original_length)
        upsampled[start_idx:end_idx] = sig
    
    return upsampled


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Multi-TF Trend Pullback V14 Strategy.
    
    Signal Logic:
    1. Resample 1h data to 4h for trend filter
    2. Calculate 4h Supertrend for major trend direction
    3. Upsample 4h trend signal to 1h timeframe
    4. On 1h: Wait for pullback to EMA-20 in direction of 4h trend
    5. Confirm with RSI (not at extremes)
    6. Confirm with volume (>= average)
    7. Regime filter: BB width percentile (avoid high vol chop)
    8. Smooth signals and apply hysteresis
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, ...]
    
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
        open_time = prices["open_time"].values if "open_time" in prices.columns else None
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
    high = np.maximum(high, close)
    low = np.minimum(low, close)
    
    # Calculate 1h indicators
    atr_1h = calculate_atr(high, low, close, ATR_PERIOD)
    ema_20_1h = calculate_ema(close, EMA_PULLBACK)
    ema_9_1h = calculate_ema(close, EMA_FAST)
    ema_21_1h = calculate_ema(close, EMA_SLOW)
    rsi_1h = calculate_rsi(close, RSI_PERIOD)
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    volume_ratio_1h = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Calculate BB width percentile for regime filter
    bb_width_percentile = np.zeros(n, dtype=np.float64)
    valid_bb = bb_width[BB_PERIOD:] > 0
    if np.any(valid_bb):
        for i in range(BB_PERIOD, n):
            if bb_width[i] > 0:
                # Calculate rolling percentile
                start_idx = max(BB_PERIOD, i - 100)
                rolling_width = bb_width[start_idx:i + 1]
                bb_width_percentile[i] = np.sum(rolling_width <= bb_width[i]) / len(rolling_width) * 100.0
    
    # Resample to 4h for trend filter
    try:
        prices_4h = resample_to_4h(prices)
        
        if len(prices_4h) < SUPERTREND_ATR_PERIOD + 5:
            # Not enough 4h data, use 1h Supertrend as fallback
            st_4h_direction = np.zeros(n, dtype=np.float64)
            _, st_4h_direction = calculate_supertrend(high, low, close, 
                                                       SUPERTREND_ATR_PERIOD, 
                                                       SUPERTREND_MULTIPLIER)
        else:
            close_4h = prices_4h["close"].values.astype(np.float64)
            high_4h = prices_4h["high"].values.astype(np.float64)
            low_4h = prices_4h["low"].values.astype(np.float64)
            
            close_4h = np.nan_to_num(close_4h, nan=0.0)
            high_4h = np.nan_to_num(high_4h, nan=0.0)
            low_4h = np.nan_to_num(low_4h, nan=0.0)
            
            close_4h = np.where(close_4h <= 0, 1.0, close_4h)
            high_4h = np.maximum(high_4h, close_4h)
            low_4h = np.minimum(low_4h, close_4h)
            
            _, st_4h_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h,
                                                          SUPERTREND_ATR_PERIOD,
                                                          SUPERTREND_MULTIPLIER)
            
            # Upsample 4h direction to 1h
            st_4h_direction = upsample_4h_signal_to_1h(st_4h_direction_4h, n, SUPERTREND_4H_BARS)
    except Exception:
        # Fallback to 1h Supertrend if resampling fails
        _, st_4h_direction = calculate_supertrend(high, low, close,
                                                   SUPERTREND_ATR_PERIOD,
                                                   SUPERTREND_MULTIPLIER)
    
    # Calculate minimum valid index
    min_valid_index = max(
        SUPERTREND_ATR_PERIOD + 2,
        ATR_PERIOD + 1,
        EMA_PULLBACK,
        RSI_PERIOD + 1,
        BB_PERIOD,
        VOLUME_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    hold_counter = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr_1h[i] <= 0 or ema_20_1h[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            hold_counter = 0
            continue
        
        # Regime filter: avoid high volatility chop (BB width > 70th percentile)
        if bb_width_percentile[i] > BB_WIDTH_HIGH_PERCENTILE:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            hold_counter = 0
            continue
        
        # Volume filter
        if volume_ratio_1h[i] < VOLUME_MIN_RATIO:
            # Can still hold position, just don't enter new
            if prev_direction != 0 and hold_counter >= HOLDBARS_MIN:
                signals[i] = prev_signal
                hold_counter += 1
            else:
                signals[i] = 0.0
                prev_signal = 0.0
                prev_direction = 0
                hold_counter = 0
            continue
        
        # Get 4h trend direction
        trend_4h = st_4h_direction[i]
        
        # Calculate pullback depth
        pullback_long = (close[i] - ema_20_1h[i]) / close[i] if close[i] > 0 else 0
        pullback_short = (ema_20_1h[i] - close[i]) / close[i] if close[i] > 0 else 0
        
        # Determine signal direction
        signal_direction = 0.0
        
        if trend_4h > 0:
            # 4h bullish: look for long pullback entries
            if pullback_long < 0.02 and pullback_long > -0.03:  # Pullback to EMA-20
                if rsi_1h[i] > RSI_LONG_MIN and rsi_1h[i] < RSI_LONG_MAX:
                    signal_direction = 1.0
        elif trend_4h < 0:
            # 4h bearish: look for short pullback entries
            if pullback_short < 0.02 and pullback_short > -0.03:  # Pullback to EMA-20
                if rsi_1h[i] > RSI_SHORT_MIN and rsi_1h[i] < RSI_SHORT_MAX:
                    signal_direction = -1.0
        
        if signal_direction == 0:
            # No new signal, maintain position if holding
            if prev_direction != 0 and hold_counter >= HOLDBARS_MIN:
                # Check if trend still valid
                if (prev_direction > 0 and trend_4h > 0) or (prev_direction < 0 and trend_4h < 0):
                    signals[i] = prev_signal
                    hold_counter += 1
                else:
                    signals[i] = 0.0
                    prev_signal = 0.0
                    prev_direction = 0
                    hold_counter = 0
            else:
                signals[i] = 0.0
                prev_signal = 0.0
                prev_direction = 0
                hold_counter = 0
            continue
        
        # Calculate signal strength
        atr_pct = atr_1h[i] / close[i]
        rsi_strength = 1.0 - abs(rsi_1h[i] - 50) / 50  # Strongest at RSI=50
        pullback_strength = 1.0 - abs(pullback_long if signal_direction > 0 else pullback_short) * 20
        
        raw_signal = signal_direction * rsi_strength * pullback_strength
        raw_signal = np.clip(raw_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        # Signal smoothing
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # Apply minimum magnitude
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Check for direction change
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction and prev_direction != 0:
            # Require stronger signal to flip
            if abs(smoothed_signal) < 0.5:
                smoothed_signal = prev_signal
        
        signals[i] = smoothed_signal
        
        if smoothed_signal != 0:
            prev_signal = smoothed_signal
            prev_direction = int(np.sign(smoothed_signal))
            hold_counter = 1
        else:
            prev_signal = 0.0
            prev_direction = 0
            hold_counter = 0
    
    # Ensure no NaN or Inf
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
    signals = np.clip(signals, -1.0, 1.0)
    
    return signals