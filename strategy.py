#!/usr/bin/env python3
"""
strategy.py - Mean Reversion Trend Filter V24
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Mean reversion with trend filter works better than pure trend following:
    - Primary signal: RSI mean reversion (oversold → long, overbought → short)
    - Trend filter: Only take mean reversion in direction of 200 EMA trend
    - Funding overlay: Extreme funding reinforces contrarian entries
    - Volatility filter: Avoid trading during extreme volatility regimes
    
    Why this works:
    - Crypto markets mean-revert after strong moves (RSI extremes)
    - Trend filter prevents fighting the major trend
    - Funding extremes indicate crowded positions ready to reverse
    - Simpler logic = more robust, fewer conflicting signals
    
    Changes from v12:
    - Simpler RSI-based primary signal (not EMA crossover)
    - Less aggressive filtering (ensure actual trades occur)
    - Lower hysteresis threshold for more trade frequency
    - Direct funding contrarian signal (not complex overlay)

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

name = "mean_reversion_trend_v24"
timeframe = "1h"
leverage = 2.5  # Moderate leverage for mean reversion

# RSI configuration for mean reversion
RSI_PERIOD = 14
RSI_OVERSOLD = 30  # Buy when RSI below this
RSI_OVERBOUGHT = 70  # Sell when RSI above this
RSI_NEUTRAL = 50  # Center point

# Trend filter configuration
EMA_TREND = 200  # Major trend filter
EMA_FAST = 50  # Secondary trend confirmation

# Funding rate configuration
FUNDING_EXTREME = 0.0008  # 0.08% per 8hr = extreme
FUNDING_MODERATE = 0.0003  # 0.03% per 8hr = moderate
FUNDING_LOOKBACK = 80  # For calculating extremes
FUNDING_WEIGHT = 0.35  # Weight of funding in signal

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_MIN = 0.002  # Minimum ATR % to trade
VOLATILITY_MAX = 0.080  # Maximum ATR % to trade

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.10  # Lower threshold for more trades
MAX_SIGNAL = 0.90  # Maximum signal magnitude
SMOOTHING_FACTOR = 0.30  # Light smoothing on signals
HYSTERESIS_THRESHOLD = 0.05  # Lower hysteresis for more trades


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


def calculate_funding_percentiles(funding_rate: np.ndarray, lookback: int = 80) -> tuple:
    """
    Calculate rolling percentile extremes of funding rate.
    Returns: (rolling_80th_percentile, rolling_20th_percentile)
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    rolling_high = np.zeros(n, dtype=np.float64)
    rolling_low = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return rolling_high, rolling_low
    
    funding_series = pd.Series(funding_rate)
    rolling_high_series = funding_series.rolling(window=lookback, min_periods=lookback).quantile(0.80)
    rolling_low_series = funding_series.rolling(window=lookback, min_periods=lookback).quantile(0.20)
    
    rolling_high = np.nan_to_num(rolling_high_series.values, nan=0.0)
    rolling_low = np.nan_to_num(rolling_low_series.values, nan=0.0)
    
    return rolling_high, rolling_low


def calculate_funding_contrarian(funding_rate: np.ndarray, 
                                  funding_high: np.ndarray,
                                  funding_low: np.ndarray,
                                  extreme_threshold: float = 0.0008,
                                  moderate_threshold: float = 0.0003,
                                  weight: float = 0.35) -> np.ndarray:
    """
    Calculate funding rate contrarian signal.
    Extreme positive funding → short bias (negative signal)
    Extreme negative funding → long bias (positive signal)
    Returns value in [-weight, weight].
    Only uses current/past funding rate (no look-ahead).
    """
    n = len(funding_rate)
    signal = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        fr = funding_rate[i]
        
        # Simple contrarian: positive funding → short, negative → long
        if fr > extreme_threshold:
            # Extreme positive funding → strong short
            signal[i] = -weight * min(1.5, fr / extreme_threshold)
        elif fr < -extreme_threshold:
            # Extreme negative funding → strong long
            signal[i] = weight * min(1.5, abs(fr) / extreme_threshold)
        elif fr > moderate_threshold:
            # Moderate positive funding → mild short
            signal[i] = -weight * 0.4 * (fr / moderate_threshold)
        elif fr < -moderate_threshold:
            # Moderate negative funding → mild long
            signal[i] = weight * 0.4 * (abs(fr) / moderate_threshold)
        else:
            signal[i] = 0.0
    
    return np.clip(signal, -weight * 1.5, weight * 1.5)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Mean Reversion Trend Filter V24 Strategy.
    
    Signal Logic:
    1. Calculate RSI for mean reversion signal
    2. Calculate trend direction from 200 EMA
    3. Generate RSI mean reversion signal (oversold→long, overbought→short)
    4. Apply trend filter (only trade with major trend)
    5. Add funding contrarian overlay
    6. Apply volatility filter
    7. Smooth and apply hysteresis
    
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
    ema_trend = calculate_ema(close, EMA_TREND)
    ema_fast = calculate_ema(close, EMA_FAST)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    funding_high, funding_low = calculate_funding_percentiles(funding_rate, FUNDING_LOOKBACK)
    funding_signal = calculate_funding_contrarian(
        funding_rate, funding_high, funding_low,
        FUNDING_EXTREME, FUNDING_MODERATE, FUNDING_WEIGHT
    )
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_TREND,
        EMA_FAST,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        FUNDING_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0 or ema_trend[i] <= 0:
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
        
        # Determine major trend direction
        trend_direction = np.sign(close[i] - ema_trend[i])
        
        # Secondary trend confirmation (50 EMA vs 200 EMA)
        if ema_fast[i] > ema_trend[i]:
            trend_confirmed = 1
        elif ema_fast[i] < ema_trend[i]:
            trend_confirmed = -1
        else:
            trend_confirmed = 0
        
        # Calculate RSI mean reversion signal
        rsi_signal = 0.0
        if rsi[i] < RSI_OVERSOLD:
            # Oversold → long signal
            rsi_signal = (RSI_OVERSOLD - rsi[i]) / RSI_OVERSOLD
            rsi_signal = min(1.0, rsi_signal)
        elif rsi[i] > RSI_OVERBOUGHT:
            # Overbought → short signal
            rsi_signal = -(rsi[i] - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT)
            rsi_signal = max(-1.0, rsi_signal)
        else:
            # Neutral RSI → weak signal based on distance from 50
            rsi_signal = (rsi[i] - RSI_NEUTRAL) / RSI_NEUTRAL * 0.3
        
        # Apply trend filter: only take mean reversion WITH the trend
        # If trend is up, prefer long signals; if trend is down, prefer short signals
        if trend_direction > 0:
            # Uptrend: favor longs, reduce short signals
            if rsi_signal < 0:
                rsi_signal *= 0.3  # Reduce short signal strength
        elif trend_direction < 0:
            # Downtrend: favor shorts, reduce long signals
            if rsi_signal > 0:
                rsi_signal *= 0.3  # Reduce long signal strength
        
        # Combine RSI signal with funding contrarian overlay
        # Funding is contrarian: extreme positive funding → short bias
        raw_signal = rsi_signal * 0.70 + funding_signal[i] * 0.30
        
        # Reinforce if both signals agree
        if np.sign(rsi_signal) == np.sign(funding_signal[i]) and abs(funding_signal[i]) > 0.1:
            raw_signal *= 1.15
        
        # Volatility normalization (slight scaling)
        vol_factor = 1.0
        if atr_pct > 0.04:
            vol_factor = 0.8  # Reduce signal in high volatility
        elif atr_pct < 0.005:
            vol_factor = 0.9  # Slight reduction in very low volatility
        raw_signal *= vol_factor
        
        # Signal smoothing (light EMA on signals)
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