#!/usr/bin/env python3
"""
strategy.py - Clean Trend 4h V14
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simplified trend-following with better risk management:
    - Primary signal: EMA crossover (20/50) for trend direction
    - Trend filter: Price above/below 200 EMA confirms major trend
    - Entry timing: RSI in neutral zone (40-60) for better entries
    - Volatility scaling: Reduce position size in high volatility
    - Funding overlay: Mild contrarian bias on extremes only
    - Hysteresis: Prevent rapid signal flipping
    
    Why this works:
    - Simpler logic = more robust across different symbols
    - 4h timeframe captures sustained trends without noise
    - Volatility scaling controls drawdown during chaotic periods
    - Conservative approach prioritizes risk-adjusted returns

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

name = "clean_trend_4h_v14"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for drawdown control

# EMA configuration for trend detection
EMA_FAST = 20
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_MIN = 40  # RSI must be above this for longs
RSI_LONG_MAX = 65  # RSI must be below this for longs (avoid overbought)
RSI_SHORT_MIN = 35  # RSI must be above this for shorts (avoid oversold)
RSI_SHORT_MAX = 60  # RSI must be below this for shorts

# Funding rate configuration (mild overlay only)
FUNDING_EXTREME_THRESHOLD = 0.0015  # 0.15% per 8hr = very extreme
FUNDING_WEIGHT = 0.20  # Conservative funding overlay
FUNDING_LOOKBACK = 80  # For calculating extremes

# Volatility configuration for position sizing
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.020  # Target ATR as % of price
VOLATILITY_MIN = 0.005  # Minimum ATR % to trade
VOLATILITY_MAX = 0.060  # Maximum ATR % to trade
VOL_SCALE_MIN = 0.5  # Minimum volatility scaling factor
VOL_SCALE_MAX = 1.5  # Maximum volatility scaling factor

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SMOOTHING_FACTOR = 0.40  # EMA smoothing for signals
HYSTERESIS_THRESHOLD = 0.15  # Minimum change to flip signal direction

# Volume confirmation
VOLUME_LOOKBACK = 15
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


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 15) -> np.ndarray:
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
                             lookback: int = 80,
                             extreme_threshold: float = 0.0015,
                             weight: float = 0.20) -> np.ndarray:
    """
    Calculate funding rate contrarian signal.
    Extreme positive funding → short bias (negative signal)
    Extreme negative funding → long bias (positive signal)
    Returns value in [-weight, weight].
    Only uses current/past funding rate (no look-ahead).
    """
    n = len(funding_rate)
    signal = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return signal
    
    funding_series = pd.Series(funding_rate)
    rolling_high = funding_series.rolling(window=lookback, min_periods=lookback).quantile(0.90)
    rolling_low = funding_series.rolling(window=lookback, min_periods=lookback).quantile(0.10)
    
    for i in range(lookback, n):
        fr = funding_rate[i]
        
        # Only act on very extreme funding rates
        if fr > extreme_threshold:
            # Strong short bias
            signal[i] = -weight * min(1.0, fr / extreme_threshold)
        elif fr < -extreme_threshold:
            # Strong long bias
            signal[i] = weight * min(1.0, abs(fr) / extreme_threshold)
        else:
            signal[i] = 0.0
    
    return signal


def calculate_trend_signal(close: np.ndarray, 
                           ema_fast: np.ndarray,
                           ema_slow: np.ndarray,
                           ema_major: np.ndarray,
                           rsi: np.ndarray) -> np.ndarray:
    """
    Calculate trend-following signal based on EMA crossover and RSI.
    Returns value in [-1, 1].
    Only uses current/past data (no look-ahead).
    """
    n = len(close)
    signal = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if close[i] <= 0 or ema_major[i] <= 0:
            signal[i] = 0.0
            continue
        
        # Primary trend direction from EMA crossover
        ema_diff_pct = (ema_fast[i] - ema_slow[i]) / close[i]
        ema_direction = np.sign(ema_diff_pct)
        
        # Major trend filter (price vs 200 EMA)
        major_filter = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend
        if ema_direction != major_filter and abs(ema_direction) > 0:
            # Conflicting signals → skip trade
            signal[i] = 0.0
            continue
        
        # Calculate trend strength (normalized)
        trend_strength = min(1.0, abs(ema_diff_pct) * 100)
        
        # RSI entry filter
        rsi_ok = False
        if ema_direction > 0:
            # Long: RSI in neutral zone (not overbought)
            if RSI_LONG_MIN <= rsi[i] <= RSI_LONG_MAX:
                rsi_ok = True
        elif ema_direction < 0:
            # Short: RSI in neutral zone (not oversold)
            if RSI_SHORT_MIN <= rsi[i] <= RSI_SHORT_MAX:
                rsi_ok = True
        
        if not rsi_ok:
            signal[i] = 0.0
            continue
        
        # Calculate final signal
        signal[i] = ema_direction * trend_strength
    
    return np.clip(signal, -1.0, 1.0)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Clean Trend 4h V14 Strategy.
    
    Signal Logic:
    1. Calculate trend signal from EMA crossover (20/50) with 200 EMA filter
    2. Apply RSI entry filter (neutral zone only)
    3. Add mild funding rate contrarian overlay
    4. Apply volatility-based position sizing
    5. Smooth signals and apply hysteresis
    6. Filter by minimum signal magnitude
    
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
    funding_signal = calculate_funding_signal(
        funding_rate, FUNDING_LOOKBACK,
        FUNDING_EXTREME_THRESHOLD, FUNDING_WEIGHT
    )
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        FUNDING_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0:
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
        
        # Calculate trend signal
        trend_sig = calculate_trend_signal(
            close, ema_fast, ema_slow, ema_major, rsi
        )[i]
        
        # Get funding overlay
        fund_sig = funding_signal[i]
        
        # Combine signals: trend is primary, funding is mild overlay
        if abs(trend_sig) > 0.1:
            # If trend and funding conflict, reduce signal strength slightly
            if np.sign(trend_sig) != np.sign(fund_sig) and abs(fund_sig) > 0.05:
                raw_signal = trend_sig * 0.85 + fund_sig * 0.15
            else:
                raw_signal = trend_sig * 0.90 + fund_sig * 0.10
        else:
            raw_signal = fund_sig  # Only funding signal if no trend
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, VOL_SCALE_MIN, VOL_SCALE_MAX)
        raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
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