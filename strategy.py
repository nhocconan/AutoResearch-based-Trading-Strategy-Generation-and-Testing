#!/usr/bin/env python3
"""
strategy.py - KAMA ADX Trend Filter V13
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

Strategy Hypothesis:
    KAMA (Kaufman Adaptive Moving Average) + ADX trend strength filter on 4h.
    
    Why this should beat Supertrend_4h_v1 (Sharpe=0.253, DD=-84.5%):
    - KAMA adapts to market efficiency ratio, reducing whipsaws in choppy markets
    - ADX > 25 filter ensures we only trade when trend is strong (avoids chop)
    - 4h timeframe captures major moves while filtering noise
    - ATR-based position sizing normalizes risk across volatility regimes
    - Volume confirmation ensures liquidity for entries
    
    Key differences from failed strategies:
    - Not pure trend (Supertrend failed with -84% DD)
    - Not mean reversion (BB/KC, Z-score all failed)
    - Adaptive MA responds to market regime changes
    - ADX filter prevents trading in low-trend environments
    
    Expected improvements:
    - Lower drawdown via ADX filter (skip choppy periods)
    - Better Sharpe via fewer but higher-quality trades
    - Works across BTC/ETH/SOL due to adaptive nature
"""

import numpy as np
import pandas as pd

# =============================================================================
# Strategy Configuration
# =============================================================================

name = "kama_adx_trend_filter_v13"
timeframe = "4h"
leverage = 1.5  # Conservative to control drawdown

# KAMA configuration (Kaufman Adaptive Moving Average)
KAMA_FAST = 2  # Fast SC period
KAMA_SLOW = 30  # Slow SC period
KAMA_ER_LOOKBACK = 10  # Efficiency Ratio lookback

# ADX configuration for trend strength filter
ADX_PERIOD = 14
ADX_MIN_THRESHOLD = 25  # Only trade when ADX > this (strong trend)
ADX_DI_DIFF_MIN = 5  # Minimum DI+ vs DI- difference

# Trend confirmation
TREND_MA_PERIOD = 50  # Major trend filter MA

# RSI for entry timing (avoid extreme entries)
RSI_PERIOD = 14
RSI_LONG_MIN = 35  # Don't long if RSI too low
RSI_SHORT_MAX = 65  # Don't short if RSI too high
RSI_OVERBOUGHT = 75
RSI_OVERSOLD = 25

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.8  # Volume must be at least this % of average

# Volatility filtering
ATR_PERIOD = 14
ATR_MIN_PCT = 0.005  # Minimum ATR % to trade
ATR_MAX_PCT = 0.08  # Maximum ATR % to trade

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.4  # EMA smoothing factor for signals
DIRECTION_CHANGE_MIN = 0.15  # Minimum change to flip direction


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_kama(close: np.ndarray, er_lookback: int = 10, 
                   fast_sc: int = 2, slow_sc: int = 30) -> np.ndarray:
    """
    Calculate Kaufman Adaptive Moving Average.
    KAMA adapts to market noise via Efficiency Ratio.
    Only uses past data (no look-ahead).
    
    Formula:
    - ER = |Close - Close[n]| / Sum(|Close[i] - Close[i-1]|)
    - SC = (ER * (fast_sc - slow_sc) + slow_sc)^2
    - KAMA = KAMA[prev] + SC * (Close - KAMA[prev])
    """
    n = len(close)
    kama = np.zeros(n, dtype=np.float64)
    
    if n < er_lookback + 1:
        return kama
    
    # Initialize first KAMA as SMA of first er_lookback periods
    kama[er_lookback] = np.mean(close[:er_lookback + 1])
    
    for i in range(er_lookback + 1, n):
        # Calculate Efficiency Ratio
        price_change = abs(close[i] - close[i - er_lookback])
        
        if price_change == 0:
            er = 0.0
        else:
            vol_sum = 0.0
            for j in range(1, er_lookback + 1):
                vol_sum += abs(close[i - j + 1] - close[i - j])
            er = price_change / vol_sum if vol_sum > 0 else 0.0
        
        # Calculate Smoothing Constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # Calculate KAMA
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_dmi(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                  period: int = 14) -> tuple:
    """
    Calculate Directional Movement Index (DMI).
    Returns: (plus_di, minus_di, adx)
    Only uses past data (no look-ahead).
    """
    n = len(close)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    tr = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    # Calculate True Range and DM for first period
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > 0 and high_diff > low_diff else 0.0
        minus_dm[i] = low_diff if low_diff > 0 and low_diff > high_diff else 0.0
        
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # Smooth DM and TR using Wilder's method
    plus_dm_smooth = np.zeros(n, dtype=np.float64)
    minus_dm_smooth = np.zeros(n, dtype=np.float64)
    tr_smooth = np.zeros(n, dtype=np.float64)
    
    # Initialize with sums for first period
    plus_dm_smooth[period] = np.sum(plus_dm[1:period + 1])
    minus_dm_smooth[period] = np.sum(minus_dm[1:period + 1])
    tr_smooth[period] = np.sum(tr[1:period + 1])
    
    # Wilder's smoothing: new = prev - prev/period + current
    for i in range(period + 1, n):
        plus_dm_smooth[i] = plus_dm_smooth[i - 1] - plus_dm_smooth[i - 1] / period + plus_dm[i]
        minus_dm_smooth[i] = minus_dm_smooth[i - 1] - minus_dm_smooth[i - 1] / period + minus_dm[i]
        tr_smooth[i] = tr_smooth[i - 1] - tr_smooth[i - 1] / period + tr[i]
    
    # Calculate DI+ and DI-
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        if tr_smooth[i] > 0:
            plus_di[i] = 100.0 * plus_dm_smooth[i] / tr_smooth[i]
            minus_di[i] = 100.0 * minus_dm_smooth[i] / tr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n, dtype=np.float64)
    adx = np.zeros(n, dtype=np.float64)
    
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100.0 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Smooth DX to get ADX
    adx[period * 2] = np.mean(dx[period:period * 2 + 1])
    for i in range(period * 2 + 1, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return plus_di, minus_di, adx


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                  period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using Wilder's smoothing.
    Only uses past data (no look-ahead).
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
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # Initialize ATR as SMA of first period TRs
    atr[period] = np.mean(tr[1:period + 1])
    
    # Wilder's smoothing
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index.
    Only uses past data (no look-ahead).
    """
    n = len(close)
    rsi = np.full(n, 50.0, dtype=np.float64)
    
    if n < period + 1:
        return rsi
    
    delta = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        delta[i] = close[i] - close[i - 1]
    
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    
    # Use Wilder's smoothing for RSI
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


def calculate_sma(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average.
    Only uses past data (no look-ahead).
    """
    n = len(close)
    sma = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return sma
    
    close_series = pd.Series(close)
    sma_values = close_series.rolling(window=period, min_periods=period).mean().values
    sma = np.nan_to_num(sma_values, nan=0.0)
    
    return sma


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


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    KAMA ADX Trend Filter V13 Strategy.
    
    Signal Logic:
    1. Calculate KAMA (adaptive MA that responds to market efficiency)
    2. Calculate ADX/DMI for trend strength and direction
    3. Filter: Only trade when ADX > 25 (strong trend)
    4. Filter: DI+ vs DI- difference > 5 (clear direction)
    5. Confirm: Price vs 50-period MA for major trend
    6. Entry timing: RSI not at extremes
    7. Volume confirmation: Volume >= 80% of average
    8. Volatility filter: ATR within reasonable bounds
    9. Smooth signals and apply hysteresis
    
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
    kama = calculate_kama(close, KAMA_ER_LOOKBACK, KAMA_FAST, KAMA_SLOW)
    plus_di, minus_di, adx = calculate_dmi(high, low, close, ADX_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    rsi = calculate_rsi(close, RSI_PERIOD)
    trend_ma = calculate_sma(close, TREND_MA_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        KAMA_ER_LOOKBACK + 2,
        ADX_PERIOD * 2 + 1,  # ADX needs 2x period for smoothing
        ATR_PERIOD + 1,
        RSI_PERIOD + 1,
        TREND_MA_PERIOD,
        VOLUME_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0 or adx[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volatility filter (not too low, not too high)
        atr_pct = atr[i] / close[i]
        if atr_pct < ATR_MIN_PCT or atr_pct > ATR_MAX_PCT:
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
        
        # ADX trend strength filter (CRITICAL - only trade strong trends)
        if adx[i] < ADX_MIN_THRESHOLD:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # DMI direction filter
        di_diff = plus_di[i] - minus_di[i]
        
        # Major trend filter (price vs 50 MA)
        major_trend = np.sign(close[i] - trend_ma[i])
        
        # Determine signal direction
        signal_direction = 0.0
        
        if di_diff > ADX_DI_DIFF_MIN:
            # DI+ > DI- → bullish
            if major_trend >= 0:  # Confirm with major trend
                # RSI entry timing
                if rsi[i] > RSI_LONG_MIN and rsi[i] < RSI_OVERBOUGHT:
                    signal_direction = 1.0
        elif di_diff < -ADX_DI_DIFF_MIN:
            # DI- > DI+ → bearish
            if major_trend <= 0:  # Confirm with major trend
                # RSI entry timing
                if rsi[i] < RSI_SHORT_MAX and rsi[i] > RSI_OVERSOLD:
                    signal_direction = -1.0
        
        if signal_direction == 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Calculate signal strength based on KAMA distance and ADX
        kama_diff = (close[i] - kama[i]) / close[i]
        
        # Normalize KAMA distance (typical range is 0-5%)
        kama_strength = np.clip(abs(kama_diff) * 20, 0.3, 1.0)
        
        # ADX strength (25-50 range normalized)
        adx_strength = np.clip((adx[i] - 20) / 30, 0.3, 1.0)
        
        # DI difference strength
        di_strength = np.clip(abs(di_diff) / 20, 0.3, 1.0)
        
        # Combine strengths
        raw_signal = signal_direction * kama_strength * adx_strength * di_strength
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # Hysteresis: don't flip direction on small changes
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < DIRECTION_CHANGE_MIN:
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