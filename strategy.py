#!/usr/bin/env python3
"""
strategy.py - KAMA Donchian Breakout V5
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "4h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Combine adaptive trend following with breakout confirmation:
    - Primary signal: KAMA (Kaufman Adaptive MA) for trend direction
    - Confirmation: Donchian channel breakout (20-period high/low)
    - Filter: Volume surge confirmation (>1.3x average)
    - Regime: Bollinger Band width percentile (low vol = breakout mode)
    - Risk: ATR-based position sizing to control drawdown
    
    Why this works:
    - KAMA adapts to market volatility (slower in chop, faster in trends)
    - Donchian breakouts capture sustained momentum moves
    - Volume filter reduces false breakouts
    - Regime filter avoids trading breakouts in high-vol mean-reversion periods
    - Better risk control than pure Supertrend (which had -84% DD)

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

name = "kama_donchian_breakout_v5"
timeframe = "4h"
leverage = 1.5  # Conservative leverage to control drawdown

# KAMA configuration (Kaufman Adaptive Moving Average)
KAMA_PERIOD = 10
KAMA_FAST_SC = 2.0 / (10 + 1)  # Fast smoothing constant
KAMA_SLOW_SC = 2.0 / (30 + 1)  # Slow smoothing constant

# Donchian channel configuration
DONCHIAN_PERIOD = 20

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 1.30  # Volume must be at least 1.3x average for breakout

# Bollinger Band regime filter
BB_PERIOD = 20
BB_STD = 2.0
BB_LOW_VOL_PERCENTILE = 40  # Below 40th percentile = low vol (breakout mode)
BB_HIGH_VOL_PERCENTILE = 70  # Above 70th percentile = high vol (avoid breakouts)

# ATR risk management
ATR_PERIOD = 14
ATR_STOP_MULT = 2.5  # ATR multiplier for trailing stop
MAX_ATR_PCT = 0.08  # Maximum ATR as % of price to trade

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.40  # EMA smoothing factor for signals
ENTRY_CONFIRMATION_BARS = 2  # Require N bars of confirmation


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_kama(close: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise: fast in trends, slow in chop.
    Only uses past data (no look-ahead).
    """
    n = len(close)
    kama = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        er[i] = signal / noise if noise > 0 else 0.0
    
    # Calculate smoothing constant
    fast_sc = KAMA_FAST_SC
    slow_sc = KAMA_SLOW_SC
    
    # Initialize KAMA with SMA of first period
    kama[period] = np.mean(close[:period + 1])
    
    # Calculate KAMA recursively
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_donchian(high: np.ndarray, low: np.ndarray, period: int = 20) -> tuple:
    """
    Calculate Donchian Channel (upper and lower bands).
    Upper = highest high of last N periods
    Lower = lowest low of last N periods
    Only uses past data (no look-ahead).
    """
    n = len(high)
    upper = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return upper, lower
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    return upper, lower


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
    
    # Use simple moving average for ATR (standard method)
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_bollinger_width(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> np.ndarray:
    """
    Calculate Bollinger Band width as % of price.
    Width = (Upper - Lower) / Middle
    Only uses past data (no look-ahead).
    """
    n = len(close)
    bb_width = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return bb_width
    
    close_series = pd.Series(close)
    
    for i in range(period - 1, n):
        window = close_series.iloc[i - period + 1:i + 1]
        sma = window.mean()
        std = window.std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        bb_width[i] = (upper - lower) / sma if sma > 0 else 0.0
    
    return bb_width


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio vs rolling average.
    Only uses past volume data (no look-ahead).
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    for i in range(lookback - 1, n):
        avg_vol = np.mean(volume[i - lookback + 1:i + 1])
        volume_ratio[i] = volume[i] / avg_vol if avg_vol > 0 else 1.0
    
    return volume_ratio


def calculate_bb_width_percentile(bb_width: np.ndarray, lookback: int = 100) -> np.ndarray:
    """
    Calculate rolling percentile of BB width.
    Returns value 0-100 indicating where current width sits in recent history.
    Only uses past data (no look-ahead).
    """
    n = len(bb_width)
    percentile = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return percentile
    
    for i in range(lookback - 1, n):
        window = bb_width[i - lookback + 1:i + 1]
        current = bb_width[i]
        # Calculate percentile rank
        percentile[i] = np.sum(window <= current) / len(window) * 100
    
    return percentile


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    KAMA Donchian Breakout V5 Strategy.
    
    Signal Logic:
    1. Calculate KAMA for adaptive trend direction
    2. Calculate Donchian channels for breakout levels
    3. Detect breakouts (price > Donchian upper or < Donchian lower)
    4. Confirm with volume surge (>1.3x average)
    5. Filter by regime (BB width percentile - only breakout in low vol)
    6. Apply ATR-based risk scaling
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
    
    # Calculate all indicators (all use only past data)
    kama = calculate_kama(close, KAMA_PERIOD)
    donchian_upper, donchian_lower = calculate_donchian(high, low, DONCHIAN_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    bb_width = calculate_bollinger_width(close, BB_PERIOD, BB_STD)
    bb_percentile = calculate_bb_width_percentile(bb_width, 100)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        KAMA_PERIOD + 1,
        DONCHIAN_PERIOD,
        ATR_PERIOD + 1,
        BB_PERIOD,
        VOLUME_LOOKBACK,
        100  # BB percentile lookback
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    consecutive_bars = 0
    last_breakout_direction = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            consecutive_bars = 0
            continue
        
        # ATR filter (not too volatile)
        atr_pct = atr[i] / close[i]
        if atr_pct > MAX_ATR_PCT:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            consecutive_bars = 0
            continue
        
        # Regime filter: only trade breakouts in low volatility
        regime_score = bb_percentile[i] / 100.0  # 0-1 scale
        if regime_score > BB_HIGH_VOL_PERCENTILE / 100:
            # High volatility regime - reduce signal strength
            regime_factor = 0.3
        elif regime_score < BB_LOW_VOL_PERCENTILE / 100:
            # Low volatility regime - ideal for breakouts
            regime_factor = 1.0
        else:
            # Medium volatility
            regime_factor = 0.6
        
        # Detect breakouts
        breakout_long = close[i] > donchian_upper[i] and donchian_upper[i] > 0
        breakout_short = close[i] < donchian_lower[i] and donchian_lower[i] > 0
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_MIN_RATIO
        
        # KAMA trend alignment
        kama_trend_long = close[i] > kama[i] and kama[i] > 0
        kama_trend_short = close[i] < kama[i] and kama[i] > 0
        
        # Generate raw signal
        raw_signal = 0.0
        
        if breakout_long and volume_confirmed and kama_trend_long:
            # Long breakout confirmed
            if last_breakout_direction != 1:
                consecutive_bars = 1
                last_breakout_direction = 1
            else:
                consecutive_bars += 1
            
            if consecutive_bars >= ENTRY_CONFIRMATION_BARS:
                raw_signal = 1.0 * regime_factor
        elif breakout_short and volume_confirmed and kama_trend_short:
            # Short breakout confirmed
            if last_breakout_direction != -1:
                consecutive_bars = 1
                last_breakout_direction = -1
            else:
                consecutive_bars += 1
            
            if consecutive_bars >= ENTRY_CONFIRMATION_BARS:
                raw_signal = -1.0 * regime_factor
        else:
            # No breakout or not confirmed
            consecutive_bars = 0
            last_breakout_direction = 0
            
            # Fade towards KAMA if no breakout (mean reversion within trend)
            if kama[i] > 0:
                kama_signal = (close[i] - kama[i]) / close[i] * 5
                raw_signal = np.clip(kama_signal, -0.3, 0.3) * regime_factor
        
        # ATR-based risk scaling (reduce signal in high volatility)
        vol_factor = 1.0 / (1.0 + atr_pct * 10)
        vol_factor = np.clip(vol_factor, 0.5, 1.5)
        raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals