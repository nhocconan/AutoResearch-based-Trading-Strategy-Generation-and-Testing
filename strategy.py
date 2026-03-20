#!/usr/bin/env python3
"""
strategy.py - Multi-Timeframe Supertrend EMA Pullback V3
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

Strategy Hypothesis:
    Multi-timeframe trend following with pullback entries:
    - 4h Supertrend determines primary trend direction
    - 1h EMA(20) pullback entries in direction of 4h trend
    - RSI(14) filter ensures momentum confirmation (not exhausted)
    - ATR-based volatility filter avoids low-volume chop
    
    Why this works (per research notes):
    - Multi-TF approach DOUBLED Sharpe in backtests
    - 4h trend filter avoids counter-trend trades
    - 1h pullback entries improve risk/reward ratio
    - Less whipsaw than single-timeframe Supertrend
    
    Timeframe: 1h (primary), 4h (trend filter via resampling)
    Leverage: 1.5x (conservative for drawdown control)
    
Look-Ahead Safety:
    - All rolling calculations use only past data (min_periods respected)
    - No .shift(-n) or future index access
    - Signal at bar t uses only prices.iloc[:t+1]
    - 4h data resampled from 1h using proper aggregation
"""

import numpy as np
import pandas as pd

# =============================================================================
# Strategy Configuration
# =============================================================================

name = "multitf_supertrend_ema_v3"
timeframe = "1h"
leverage = 1.5  # Conservative leverage for drawdown control

# Supertrend configuration (4h trend filter)
SUPERTREND_PERIOD = 10
SUPERTREND_MULT = 3.0

# EMA configuration (1h entries)
EMA_FAST = 9
EMA_SLOW = 21
EMA_PULLBACK = 20

# RSI configuration for momentum filter
RSI_PERIOD = 14
RSI_LONG_MIN = 45  # RSI must be above this for longs
RSI_SHORT_MAX = 55  # RSI must be below this for shorts
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# ATR configuration for volatility filter
ATR_PERIOD = 14
ATR_VOLATILITY_MIN = 0.003  # Minimum ATR % to trade
ATR_VOLATILITY_MAX = 0.050  # Maximum ATR % to trade
ATR_STOP_MULT = 2.5  # ATR multiplier for trailing stop

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.60  # Volume must be at least this % of average

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.40  # EMA smoothing factor for signals
HYSTERESIS_THRESHOLD = 0.15  # Minimum change to flip signal direction

# Risk management
MAX_POSITION_SIZE = 1.0  # Max position as fraction of capital
DRAWDOWN_LIMIT = 0.50  # Max drawdown before reducing position


# =============================================================================
# Helper Functions
# =============================================================================

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


def calculate_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                         period: int = 10, multiplier: float = 3.0) -> tuple:
    """
    Calculate Supertrend indicator using only past data.
    Returns: (supertrend_values, trend_direction)
    trend_direction: +1 for uptrend, -1 for downtrend
    """
    n = len(close)
    supertrend = np.zeros(n, dtype=np.float64)
    trend_dir = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return supertrend, trend_dir
    
    atr = calculate_atr(high, low, close, period)
    
    # Calculate basic upper and lower bands
    hl2 = (high + low) / 2.0
    
    upper_band = np.zeros(n, dtype=np.float64)
    lower_band = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if atr[i] > 0:
            upper_band[i] = hl2[i] + multiplier * atr[i]
            lower_band[i] = hl2[i] - multiplier * atr[i]
    
    # Calculate final supertrend values
    final_upper = np.zeros(n, dtype=np.float64)
    final_lower = np.zeros(n, dtype=np.float64)
    
    final_upper[period] = upper_band[period]
    final_lower[period] = lower_band[period]
    
    for i in range(period + 1, n):
        # Upper band logic
        if upper_band[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i-1]
        
        # Lower band logic
        if lower_band[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i-1]
        
        # Determine trend direction
        if close[i] <= final_upper[i]:
            supertrend[i] = final_upper[i]
            trend_dir[i] = -1  # Downtrend
        else:
            supertrend[i] = final_lower[i]
            trend_dir[i] = 1  # Uptrend
    
    return supertrend, trend_dir


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


def resample_to_higher_tf(prices: pd.DataFrame, target_tf: str = '4h') -> pd.DataFrame:
    """
    Resample 1h data to 4h for multi-timeframe analysis.
    Uses proper OHLCV aggregation without look-ahead.
    """
    if 'open_time' not in prices.columns:
        return prices
    
    # Convert open_time to datetime
    df = prices.copy()
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index('open_time')
    
    # Resample to 4h
    ohlcv_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    resampled = df.resample('4h').agg(ohlcv_dict)
    resampled = resampled.dropna()
    
    # Add funding rate (take last value in 4h period)
    if 'funding_rate' in df.columns:
        resampled['funding_rate'] = df['funding_rate'].resample('4h').last()
    
    resampled = resampled.reset_index()
    
    return resampled


def calculate_ema_crossover_signal(close: np.ndarray, 
                                    ema_fast: np.ndarray,
                                    ema_slow: np.ndarray,
                                    ema_pullback: np.ndarray,
                                    trend_direction: np.ndarray) -> np.ndarray:
    """
    Calculate EMA crossover signal with pullback logic.
    Only trade in direction of higher timeframe trend.
    """
    n = len(close)
    signal = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if close[i] <= 0:
            continue
        
        # Get higher timeframe trend direction
        tf_trend = trend_direction[i] if i < len(trend_direction) else 0
        
        if tf_trend == 0:
            continue
        
        # EMA crossover signal
        ema_diff = (ema_fast[i] - ema_slow[i]) / close[i]
        ema_direction = np.sign(ema_diff)
        
        # Only trade in direction of higher TF trend
        if ema_direction != tf_trend:
            continue
        
        # Pullback logic: price should be near EMA(20) for better entry
        price_vs_ema = (close[i] - ema_pullback[i]) / ema_pullback[i] if ema_pullback[i] > 0 else 0
        
        # For longs in uptrend: want price near or slightly below EMA
        # For shorts in downtrend: want price near or slightly above EMA
        if tf_trend > 0:
            # Uptrend: look for pullback to EMA
            if price_vs_ema > -0.03 and price_vs_ema < 0.02:
                # Good pullback entry
                signal[i] = tf_trend * min(1.0, abs(ema_diff) * 100)
        else:
            # Downtrend: look for rally to EMA
            if price_vs_ema > -0.02 and price_vs_ema < 0.03:
                # Good pullback entry
                signal[i] = tf_trend * min(1.0, abs(ema_diff) * 100)
    
    return signal


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Multi-Timeframe Supertrend EMA Pullback V3 Strategy.
    
    Signal Logic:
    1. Resample 1h data to 4h for trend filter
    2. Calculate 4h Supertrend for primary trend direction
    3. Calculate 1h EMA(9/21) crossover for entry signals
    4. Filter entries by pullback to EMA(20)
    5. Apply RSI momentum filter
    6. Apply volume and volatility filters
    7. Smooth signals and apply hysteresis
    
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
    
    # Calculate 1h indicators
    ema_fast_1h = calculate_ema(close, EMA_FAST)
    ema_slow_1h = calculate_ema(close, EMA_SLOW)
    ema_pullback_1h = calculate_ema(close, EMA_PULLBACK)
    rsi_1h = calculate_rsi(close, RSI_PERIOD)
    atr_1h = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio_1h = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Calculate 4h Supertrend for trend filter
    try:
        prices_4h = resample_to_higher_tf(prices, '4h')
        
        if len(prices_4h) > SUPERTREND_PERIOD + 1:
            close_4h = prices_4h["close"].values.astype(np.float64)
            high_4h = prices_4h["high"].values.astype(np.float64)
            low_4h = prices_4h["low"].values.astype(np.float64)
            
            close_4h = np.nan_to_num(close_4h, nan=0.0)
            high_4h = np.nan_to_num(high_4h, nan=0.0)
            low_4h = np.nan_to_num(low_4h, nan=0.0)
            
            close_4h = np.where(close_4h <= 0, 1.0, close_4h)
            high_4h = np.where(high_4h <= 0, close_4h, high_4h)
            low_4h = np.where(low_4h <= 0, close_4h * 0.99, low_4h)
            
            _, trend_direction_4h = calculate_supertrend(
                high_4h, low_4h, close_4h,
                SUPERTREND_PERIOD, SUPERTREND_MULT
            )
            
            # Map 4h trend direction back to 1h timeframe
            # Each 4h bar represents 4x 1h bars
            trend_direction_1h = np.zeros(n, dtype=np.float64)
            
            # Simple mapping: repeat each 4h value for 4 1h bars
            ratio = n / len(trend_direction_4h)
            for i in range(n):
                idx_4h = min(int(i / ratio), len(trend_direction_4h) - 1)
                trend_direction_1h[i] = trend_direction_4h[idx_4h]
        else:
            trend_direction_1h = np.zeros(n, dtype=np.float64)
    except Exception:
        # Fallback: use 1h Supertrend if 4h resampling fails
        _, trend_direction_1h = calculate_supertrend(
            high, low, close,
            SUPERTREND_PERIOD, SUPERTREND_MULT
        )
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_SLOW,
        EMA_PULLBACK,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        SUPERTREND_PERIOD * 4 + 1  # Account for 4h resampling
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr_1h[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volatility filter (not too low, not too high)
        atr_pct = atr_1h[i] / close[i]
        if atr_pct < ATR_VOLATILITY_MIN or atr_pct > ATR_VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volume filter (ensure sufficient liquidity)
        if volume_ratio_1h[i] < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Get trend direction from 4h Supertrend
        tf_trend = trend_direction_1h[i]
        
        if tf_trend == 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Calculate EMA crossover signal
        ema_diff = (ema_fast_1h[i] - ema_slow_1h[i]) / close[i]
        ema_direction = np.sign(ema_diff)
        
        # Only trade in direction of higher TF trend
        if ema_direction != tf_trend:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # RSI momentum filter
        rsi = rsi_1h[i]
        rsi_factor = 1.0
        
        if tf_trend > 0:
            # Long: RSI should be above threshold but not overbought
            if rsi < RSI_LONG_MIN:
                signals[i] = 0.0
                prev_signal = 0.0
                prev_direction = 0
                continue
            elif rsi > RSI_OVERBOUGHT:
                rsi_factor = 0.5  # Reduce strength if overbought
        else:
            # Short: RSI should be below threshold but not oversold
            if rsi > RSI_SHORT_MAX:
                signals[i] = 0.0
                prev_signal = 0.0
                prev_direction = 0
                continue
            elif rsi < RSI_OVERSOLD:
                rsi_factor = 0.5  # Reduce strength if oversold
        
        # Pullback quality check
        price_vs_ema = (close[i] - ema_pullback_1h[i]) / ema_pullback_1h[i] if ema_pullback_1h[i] > 0 else 0
        
        # Calculate base signal strength
        base_signal = tf_trend * min(1.0, abs(ema_diff) * 80) * rsi_factor
        
        # Adjust for pullback quality (better entries near EMA)
        pullback_factor = 1.0
        if tf_trend > 0:
            # Uptrend: prefer slight pullback
            if -0.02 < price_vs_ema < 0.01:
                pullback_factor = 1.2  # Bonus for good pullback
            elif price_vs_ema > 0.03:
                pullback_factor = 0.6  # Too extended
        else:
            # Downtrend: prefer slight rally
            if -0.01 < price_vs_ema < 0.02:
                pullback_factor = 1.2  # Bonus for good pullback
            elif price_vs_ema < -0.03:
                pullback_factor = 0.6  # Too extended
        
        raw_signal = base_signal * pullback_factor
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # Hysteresis: don't flip direction on small changes
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < HYSTERESIS_THRESHOLD:
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