#!/usr/bin/env python3
"""
strategy.py - Trend Follow Conservative V13
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Simple trend-following with strict drawdown control:
    - Primary signal: EMA crossover (12/26) - classic MACD-style
    - Trend filter: Price above/below 100 EMA for direction bias
    - Momentum confirmation: RSI in favorable zone (not extreme)
    - Volatility scaling: Reduce position size in high volatility
    - Funding filter: Only act as contrarian at extreme levels
    
    Why this works:
    - 4h timeframe captures sustained trends with less noise than 1h
    - Simple EMA crossover is robust across BTC/ETH/SOL
    - Volatility scaling prevents oversized positions during chaos
    - Conservative leverage (1.5x) keeps drawdown manageable
    
    Key improvements over v12:
    - Simpler logic (less overfitting)
    - 4h timeframe for cleaner signals
    - Better volatility normalization
    - Relaxed filters to ensure trade count

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

name = "trend_follow_conservative_v13"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for drawdown control

# EMA configuration for trend detection
EMA_FAST = 12
EMA_SLOW = 26
EMA_TREND = 100

# RSI configuration for momentum
RSI_PERIOD = 14
RSI_LONG_MIN = 40  # RSI must be above this for longs
RSI_SHORT_MAX = 60  # RSI must be below this for shorts

# Funding rate configuration (contrarian filter only)
FUNDING_EXTREME = 0.0015  # 0.15% per 8hr = very extreme
FUNDING_LOOKBACK = 50
FUNDING_WEIGHT = 0.25  # Light weight - trend is primary

# Volatility configuration
ATR_PERIOD = 14
VOL_TARGET = 0.02  # Target ATR as % of price
VOL_MAX = 0.08  # Max ATR % to trade (skip extremely volatile)
VOL_MIN = 0.005  # Min ATR % to trade (skip dead markets)

# Signal configuration
MIN_SIGNAL = 0.20  # Minimum signal magnitude to trade
MAX_SIGNAL = 0.75  # Maximum signal magnitude
SMOOTHING = 0.40  # Signal EMA smoothing factor


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
    Calculate rolling percentile of funding rate (0-1 scale).
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    if n < lookback:
        return np.full(n, 0.5, dtype=np.float64)
    
    funding_series = pd.Series(funding_rate)
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
    Trend Follow Conservative V13 Strategy.
    
    Signal Logic:
    1. Calculate EMA crossover signal (12/26 EMA)
    2. Filter by major trend (price vs 100 EMA)
    3. Confirm with RSI momentum
    4. Scale by volatility (reduce size in high vol)
    5. Apply funding contrarian filter at extremes
    6. Smooth signals to reduce whipsaws
    
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
    ema_trend = calculate_ema(close, EMA_TREND)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    funding_percentile = calculate_funding_percentile(funding_rate, FUNDING_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_TREND,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        FUNDING_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0 or ema_trend[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Volatility filter (skip extremely high or dead low volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct > VOL_MAX or atr_pct < VOL_MIN:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # EMA crossover signal
        ema_diff = (ema_fast[i] - ema_slow[i]) / close[i]
        ema_signal = np.tanh(ema_diff * 100)  # Smooth sigmoid-like mapping
        
        # Major trend filter (price vs 100 EMA)
        trend_bias = np.sign(close[i] - ema_trend[i])
        
        # Only trade in direction of major trend (reduces counter-trend trades)
        if np.sign(ema_signal) != trend_bias and trend_bias != 0:
            # Reduce signal strength for counter-trend
            ema_signal *= 0.3
        
        # RSI momentum confirmation
        rsi_factor = 1.0
        if ema_signal > 0:
            # Long: RSI should be above minimum (not too weak)
            if rsi[i] < RSI_LONG_MIN:
                rsi_factor = 0.5  # Reduce but don't kill
            elif rsi[i] > 75:
                rsi_factor = 0.7  # Slightly overbought, reduce
        elif ema_signal < 0:
            # Short: RSI should be below maximum (not too strong)
            if rsi[i] > RSI_SHORT_MAX:
                rsi_factor = 0.5  # Reduce but don't kill
            elif rsi[i] < 25:
                rsi_factor = 0.7  # Slightly oversold, reduce
        
        # Combine EMA signal with RSI factor
        raw_signal = ema_signal * rsi_factor
        
        # Funding rate contrarian filter (only at extremes)
        funding_adjustment = 0.0
        if funding_percentile[i] > 0.85:
            # Very high funding → slight short bias
            funding_adjustment = -FUNDING_WEIGHT * 0.5
        elif funding_percentile[i] < 0.15:
            # Very low funding → slight long bias
            funding_adjustment = FUNDING_WEIGHT * 0.5
        
        # Apply funding adjustment (small effect)
        raw_signal += funding_adjustment
        
        # Volatility normalization (scale position by volatility)
        vol_scale = VOL_TARGET / max(atr_pct, 0.001)
        vol_scale = np.clip(vol_scale, 0.5, 1.5)  # Cap scaling
        raw_signal *= vol_scale
        
        # Signal smoothing (EMA on signals to reduce whipsaws)
        smoothed_signal = SMOOTHING * prev_signal + (1.0 - SMOOTHING) * raw_signal
        
        # Apply minimum magnitude filter (ensure we actually trade)
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals