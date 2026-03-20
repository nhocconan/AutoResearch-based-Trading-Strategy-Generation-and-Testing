#!/usr/bin/env python3
"""
strategy.py - Trend Momentum V13
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "4h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    4h trend-following with RSI entry timing and volatility control:
    - Primary signal: EMA trend direction (12/26 EMA crossover)
    - Trend filter: Price above/below 100 EMA for major trend
    - Entry timing: RSI momentum (avoid overbought/oversold entries)
    - Volatility scaling: Normalize position size by ATR
    - Funding filter: Reduce position when funding is extreme
    
    Why 4h timeframe:
    - Cleaner trends than 1h/15m, fewer false signals
    - Lower transaction costs relative to signal quality
    - Better risk/reward for trend-following
    - Works well across BTC/ETH/SOL

    Why this should reduce drawdown:
    - Volatility-based position sizing reduces exposure in high vol
    - RSI entry timing avoids chasing moves
    - Funding filter prevents entering crowded trades
    - Conservative leverage (1.5x)

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

name = "trend_momentum_v13"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for drawdown control

# EMA configuration for trend detection
EMA_FAST = 12
EMA_SLOW = 26
EMA_MAJOR = 100

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_MIN = 40  # Minimum RSI for long entry
RSI_LONG_MAX = 70  # Maximum RSI for long entry (avoid overbought)
RSI_SHORT_MIN = 30  # Minimum RSI for short entry (avoid oversold)
RSI_SHORT_MAX = 60  # Maximum RSI for short entry

# Funding rate configuration
FUNDING_EXTREME = 0.0015  # 0.15% per 8hr = very extreme
FUNDING_MODERATE = 0.0005  # 0.05% per 8hr = moderate
FUNDING_LOOKBACK = 50  # For calculating extremes
FUNDING_REDUCTION = 0.50  # Reduce position by this % when funding extreme

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.020  # Target ATR as % of price
VOLATILITY_MIN = 0.005  # Minimum ATR % to trade
VOLATILITY_MAX = 0.080  # Maximum ATR % to trade

# Signal configuration
MIN_SIGNAL = 0.20  # Minimum signal magnitude to trade
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.30  # EMA smoothing factor for signals
TREND_STRENGTH_MULT = 40  # Multiplier for trend strength


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
    return np.nan_to_num(ema_values, nan=0.0)


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using only past data.
    """
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0, dtype=np.float64)
    
    close_series = pd.Series(close)
    delta = close_series.diff()
    
    gains = delta.where(delta > 0, 0.0)
    losses = (-delta).where(delta < 0, 0.0)
    
    avg_gains = gains.ewm(com=period - 1, min_periods=period).mean()
    avg_losses = losses.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gains / avg_losses.replace(0, np.inf)
    rsi_series = 100.0 - (100.0 / (1.0 + rs))
    
    return np.nan_to_num(rsi_series.values, nan=50.0)


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
    
    return np.nan_to_num(atr_series.values, nan=0.0)


def calculate_funding_percentile(funding_rate: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calculate rolling percentile of funding rate.
    Returns value in [0, 1] where 1 = highest in lookback.
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    if n < lookback:
        return np.zeros(n, dtype=np.float64)
    
    funding_series = pd.Series(funding_rate)
    # Calculate rank within rolling window
    rolling_rank = funding_series.rolling(window=lookback, min_periods=lookback).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10),
        raw=False
    )
    
    return np.nan_to_num(rolling_rank.values, nan=0.5)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Momentum V13 Strategy.
    
    Signal Logic:
    1. Calculate EMA trend direction (12/26 crossover)
    2. Filter by major trend (price vs 100 EMA)
    3. Check RSI for entry timing (avoid extremes)
    4. Scale by volatility (ATR normalization)
    5. Reduce position when funding is extreme
    6. Smooth signals and apply minimum magnitude filter
    
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
    funding_percentile = calculate_funding_percentile(funding_rate, FUNDING_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW + 5,
        RSI_PERIOD + 5,
        ATR_PERIOD + 5,
        FUNDING_LOOKBACK
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
        
        # Calculate trend direction from EMA crossover
        ema_diff = ema_fast[i] - ema_slow[i]
        ema_direction = np.sign(ema_diff)
        
        # Major trend filter (price vs 100 EMA)
        major_direction = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend
        if ema_direction != major_direction or ema_direction == 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate trend strength
        trend_strength = abs(ema_diff) / close[i] * TREND_STRENGTH_MULT
        trend_strength = np.clip(trend_strength, 0.0, 1.0)
        
        # RSI entry timing filter
        rsi_valid = False
        if ema_direction > 0:
            # Long: RSI should be in valid range (not overbought)
            if RSI_LONG_MIN <= rsi[i] <= RSI_LONG_MAX:
                rsi_valid = True
        elif ema_direction < 0:
            # Short: RSI should be in valid range (not oversold)
            if RSI_SHORT_MIN <= rsi[i] <= RSI_SHORT_MAX:
                rsi_valid = True
        
        if not rsi_valid:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Base signal from trend direction and strength
        raw_signal = ema_direction * trend_strength
        
        # Funding rate filter (reduce position when funding is extreme)
        funding_reduction = 1.0
        if funding_rate[i] > FUNDING_EXTREME and ema_direction > 0:
            # Extreme positive funding + long = crowded trade, reduce position
            funding_reduction = 1.0 - FUNDING_REDUCTION
        elif funding_rate[i] < -FUNDING_EXTREME and ema_direction < 0:
            # Extreme negative funding + short = crowded trade, reduce position
            funding_reduction = 1.0 - FUNDING_REDUCTION
        elif abs(funding_rate[i]) > FUNDING_MODERATE:
            # Moderate extreme funding = slight reduction
            funding_reduction = 0.85
        
        raw_signal *= funding_reduction
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals to reduce whipsaws)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals