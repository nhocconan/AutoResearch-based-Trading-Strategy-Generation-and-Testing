#!/usr/bin/env python3
"""
strategy.py - Volatility Adaptive Trend V16
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "4h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    4h timeframe trend-following with volatility-adaptive position sizing:
    - Primary signal: EMA crossover (20/50) for trend direction
    - Confirmation: Price above/below 200 EMA for major trend filter
    - Momentum: RSI must support trend direction (not extreme)
    - Volatility scaling: Reduce position size when ATR is high
    - Funding filter: Avoid entering when funding is extremely against position
    - Signal smoothing: EMA on signals to reduce whipsaws
    
    Why 4h timeframe:
    - Cleaner signals than 1h/15m (less noise)
    - More trades than 1d (better statistical significance)
    - Lower transaction costs relative to signal quality
    - Better for trend-following strategies
    
    Drawdown Control:
    - Volatility-based position sizing (smaller positions in high vol)
    - Conservative leverage (1.5x max)
    - Signal smoothing to avoid rapid flips
    - Minimum signal threshold to avoid weak trades

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

name = "vol_adaptive_trend_v16"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for drawdown control

# EMA configuration for trend detection
EMA_FAST = 20
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration for momentum confirmation
RSI_PERIOD = 14
RSI_LONG_MIN = 45  # RSI must be above this for longs
RSI_SHORT_MAX = 55  # RSI must be below this for shorts
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Funding rate configuration
FUNDING_EXTREME = 0.0015  # 0.15% per 8hr = extreme, avoid trading against
FUNDING_LOOKBACK = 80  # For calculating rolling extremes
FUNDING_FILTER_WEIGHT = 0.3  # How much funding affects signal

# Volatility configuration for position sizing
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.02  # Target ATR as % of price
VOLATILITY_MIN = 0.005  # Minimum ATR % to trade
VOLATILITY_MAX = 0.08  # Maximum ATR % (reduce position above this)

# Signal configuration
MIN_SIGNAL = 0.20  # Minimum signal magnitude to trade
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.4  # EMA smoothing factor for signals (0=none, 1=max)
SIGNAL_HYSTERESIS = 0.12  # Minimum change to flip direction

# Volume filter
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.6  # Volume must be at least this % of average


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
    
    atr = np.nan_to_num(atr_series.values, nan=0.0)
    
    return atr


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio vs rolling average.
    Only uses past volume data (no look-ahead).
    """
    n = len(volume)
    if n < lookback:
        return np.ones(n, dtype=np.float64)
    
    volume_series = pd.Series(volume)
    rolling_avg = volume_series.rolling(window=lookback, min_periods=lookback).mean()
    
    volume_ratio = np.nan_to_num(volume_series.values / rolling_avg.values, nan=1.0)
    
    return volume_ratio


def calculate_funding_percentile(funding_rate: np.ndarray, lookback: int = 80) -> np.ndarray:
    """
    Calculate rolling percentile rank of funding rate.
    Returns value in [0, 1] where 1 = highest in lookback period.
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    if n < lookback:
        return np.full(n, 0.5, dtype=np.float64)
    
    funding_series = pd.Series(funding_rate)
    percentile = funding_series.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10),
        raw=False
    )
    
    percentile = np.nan_to_num(percentile.values, nan=0.5)
    
    return percentile


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Volatility Adaptive Trend V16 Strategy.
    
    Signal Logic:
    1. Calculate trend direction from EMA crossover (20/50)
    2. Filter by major trend (price vs 200 EMA)
    3. Confirm with RSI momentum (not overbought/oversold)
    4. Adjust for funding rate extremes (contrarian filter)
    5. Scale by volatility (smaller positions in high vol)
    6. Smooth signals to reduce whipsaws
    7. Apply minimum magnitude threshold
    
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
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    funding_percentile = calculate_funding_percentile(funding_rate, FUNDING_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW + 5,
        RSI_PERIOD + 2,
        ATR_PERIOD + 2,
        VOLUME_LOOKBACK,
        FUNDING_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0 or ema_major[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volatility filter (not too low, not too high)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volume filter (ensure sufficient liquidity)
        if volume_ratio[i] < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Determine trend direction from EMA crossover
        ema_diff_ratio = (ema_fast[i] - ema_slow[i]) / close[i]
        ema_direction = np.sign(ema_diff_ratio)
        
        # Major trend filter (price vs 200 EMA)
        major_direction = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend (stronger filter)
        if ema_direction != 0 and ema_direction != major_direction:
            # Conflicting signals → skip trade
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # RSI momentum confirmation
        rsi_valid = False
        if ema_direction > 0:
            # Long: RSI must be above minimum but not overbought
            if RSI_LONG_MIN <= rsi[i] <= RSI_OVERBOUGHT:
                rsi_valid = True
        elif ema_direction < 0:
            # Short: RSI must be below maximum but not oversold
            if RSI_OVERSOLD <= rsi[i] <= RSI_SHORT_MAX:
                rsi_valid = True
        
        if not rsi_valid:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Calculate base signal strength from EMA separation
        base_strength = min(abs(ema_diff_ratio) * 100, 1.0)
        
        # Funding rate filter (contrarian)
        # Extreme funding against our position → reduce signal
        funding_filter = 1.0
        if ema_direction > 0 and funding_percentile[i] > 0.85:
            # Long but funding is very positive (crowded longs) → reduce
            funding_filter = 1.0 - FUNDING_FILTER_WEIGHT
        elif ema_direction < 0 and funding_percentile[i] < 0.15:
            # Short but funding is very negative (crowded shorts) → reduce
            funding_filter = 1.0 - FUNDING_FILTER_WEIGHT
        
        # Combine signals
        raw_signal = ema_direction * base_strength * funding_filter
        
        # Volatility-based position sizing
        # Higher volatility → smaller position
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.4, 1.5)  # Cap the adjustment
        raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals to reduce whipsaws)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # Hysteresis: don't flip direction on small changes
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < SIGNAL_HYSTERESIS:
                smoothed_signal = prev_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals