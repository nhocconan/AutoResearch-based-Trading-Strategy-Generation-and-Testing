#!/usr/bin/env python3
"""
strategy.py - Funding Mean Reversion V7
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "4h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Funding rate mean reversion with trend filter:
    - Primary signal: Extreme funding rates indicate crowded positions
    - When funding is very positive → shorts are crowded → long bias
    - When funding is very negative → longs are crowded → short bias
    - Trend filter: Only take signals that don't fight strong trends
    - RSI confirmation: Avoid catching falling knives
    - Volatility scaling: Reduce position size in high volatility
    
    Why this works on 4h:
    - Funding rates reset every 8h, 4h captures 2 funding cycles
    - Less noise than 1h/15m, cleaner mean reversion signals
    - Extreme funding persists long enough to capture reversal
    - Conservative approach controls drawdown

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

name = "funding_mean_reversion_v7"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for drawdown control

# EMA configuration for trend filter
EMA_TREND = 50
EMA_MAJOR = 200

# RSI configuration
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_NEUTRAL_ZONE = 45  # RSI must be above/below this for entries

# Funding rate configuration (CRITICAL for this strategy)
FUNDING_EXTREME_LONG = 0.0008  # 0.08% per 8hr = very positive
FUNDING_EXTREME_SHORT = -0.0008  # -0.08% per 8hr = very negative
FUNDING_MODERATE_LONG = 0.0003  # 0.03% per 8hr = moderate positive
FUNDING_MODERATE_SHORT = -0.0003  # -0.03% per 8hr = moderate negative
FUNDING_LOOKBACK = 50  # For calculating rolling extremes
FUNDING_MAX_SIGNAL = 0.70  # Max signal from funding alone

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.020  # Target ATR as % of price
VOLATILITY_MIN = 0.005  # Minimum ATR % to trade
VOLATILITY_MAX = 0.040  # Maximum ATR % to trade
VOLATILITY_SCALE = 0.50  # How much to scale by volatility

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.40  # EMA smoothing factor for signals
TREND_FILTER_STRENGTH = 0.30  # How much trend affects signal

# Trade management
MIN_BARS_BETWEEN_FLIPS = 3  # Minimum bars before allowing signal flip


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


def calculate_funding_percentiles(funding_rate: np.ndarray, lookback: int = 50) -> tuple:
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


def calculate_funding_signal(funding_rate: np.ndarray,
                             rolling_high: np.ndarray,
                             rolling_low: np.ndarray,
                             extreme_long: float = 0.0008,
                             extreme_short: float = -0.0008,
                             moderate_long: float = 0.0003,
                             moderate_short: float = -0.0003,
                             max_signal: float = 0.70) -> np.ndarray:
    """
    Calculate funding rate mean reversion signal.
    Extreme positive funding → short bias (negative signal) - crowded longs
    Extreme negative funding → long bias (positive signal) - crowded shorts
    Returns value in [-max_signal, max_signal].
    Only uses current/past funding rate (no look-ahead).
    """
    n = len(funding_rate)
    signal = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        fr = funding_rate[i]
        fr_high = rolling_high[i]
        fr_low = rolling_low[i]
        
        # Mean reversion: extreme funding → opposite position
        if fr > extreme_long:
            # Very positive funding → shorts are crowded → go long
            # But cap the signal
            signal[i] = min(max_signal, 0.5 + (fr - extreme_long) * 200)
        elif fr < extreme_short:
            # Very negative funding → longs are crowded → go short
            signal[i] = -min(max_signal, 0.5 + (extreme_short - fr) * 200)
        elif fr > moderate_long:
            # Moderate positive funding → mild long bias
            signal[i] = 0.2 + (fr - moderate_long) * 100
            signal[i] = min(0.4, signal[i])
        elif fr < moderate_short:
            # Moderate negative funding → mild short bias
            signal[i] = -0.2 - (moderate_short - fr) * 100
            signal[i] = max(-0.4, signal[i])
        else:
            # Neutral funding → no signal from funding alone
            signal[i] = 0.0
    
    return np.clip(signal, -max_signal, max_signal)


def calculate_trend_filter(close: np.ndarray,
                           ema_trend: np.ndarray,
                           ema_major: np.ndarray) -> np.ndarray:
    """
    Calculate trend filter based on EMA alignment.
    Returns value in [-1, 1] where positive = uptrend, negative = downtrend.
    Only uses current/past data (no look-ahead).
    """
    n = len(close)
    trend = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if ema_trend[i] <= 0 or ema_major[i] <= 0:
            trend[i] = 0.0
            continue
        
        # Price vs trend EMA
        price_vs_trend = (close[i] - ema_trend[i]) / ema_trend[i]
        
        # Trend EMA vs major EMA
        ema_vs_major = (ema_trend[i] - ema_major[i]) / ema_major[i]
        
        # Combine both signals
        trend[i] = np.sign(price_vs_trend + ema_vs_major) * min(1.0, abs(price_vs_trend) * 20 + abs(ema_vs_major) * 20)
    
    return np.clip(trend, -1.0, 1.0)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Funding Mean Reversion V7 Strategy.
    
    Signal Logic:
    1. Calculate funding mean reversion signal (primary driver)
    2. Calculate trend filter (avoid fighting strong trends)
    3. Calculate RSI (avoid extremes)
    4. Combine: funding signal adjusted by trend alignment
    5. Apply volatility scaling
    6. Smooth signals with EMA
    7. Apply minimum magnitude filter
    
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
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    funding_high, funding_low = calculate_funding_percentiles(funding_rate, FUNDING_LOOKBACK)
    funding_signal = calculate_funding_signal(
        funding_rate, funding_high, funding_low,
        FUNDING_EXTREME_LONG, FUNDING_EXTREME_SHORT,
        FUNDING_MODERATE_LONG, FUNDING_MODERATE_SHORT,
        FUNDING_MAX_SIGNAL
    )
    
    trend_filter = calculate_trend_filter(close, ema_trend, ema_major)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_TREND,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        FUNDING_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    bars_since_flip = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            bars_since_flip = 0
            continue
        
        # Volatility filter (not too low, not too high)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            bars_since_flip = 0
            continue
        
        # Get base funding signal (mean reversion)
        fund_sig = funding_signal[i]
        
        # Get trend filter
        trend = trend_filter[i]
        
        # Adjust funding signal by trend alignment
        # If funding says long but trend is strongly down, reduce signal
        # If funding and trend align, reinforce signal
        if abs(fund_sig) > 0.1:
            funding_direction = np.sign(fund_sig)
            trend_direction = np.sign(trend)
            
            if funding_direction == trend_direction:
                # Aligned → reinforce
                trend_adjustment = 1.0 + TREND_FILTER_STRENGTH * abs(trend)
            elif abs(trend) > 0.5:
                # Strong opposing trend → reduce signal significantly
                trend_adjustment = 1.0 - TREND_FILTER_STRENGTH * abs(trend)
                trend_adjustment = max(0.3, trend_adjustment)
            else:
                # Weak opposing trend → slight reduction
                trend_adjustment = 1.0 - TREND_FILTER_STRENGTH * 0.5 * abs(trend)
            
            raw_signal = fund_sig * trend_adjustment
        else:
            raw_signal = fund_sig
        
        # RSI filter: avoid extreme RSI levels for entries
        if abs(raw_signal) > 0.1:
            if raw_signal > 0:  # Long signal
                if rsi[i] > RSI_OVERBOUGHT:
                    raw_signal *= 0.5  # Reduce if overbought
                elif rsi[i] < RSI_NEUTRAL_ZONE:
                    raw_signal *= 1.2  # Boost if room to run
            else:  # Short signal
                if rsi[i] < RSI_OVERSOLD:
                    raw_signal *= 0.5  # Reduce if oversold
                elif rsi[i] > (100 - RSI_NEUTRAL_ZONE):
                    raw_signal *= 1.2  # Boost if room to run
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 1.5)
        raw_signal *= (1.0 - VOLATILITY_SCALE + VOLATILITY_SCALE * vol_factor)
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # Track signal flips
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if bars_since_flip < MIN_BARS_BETWEEN_FLIPS:
                # Too soon to flip → keep previous signal
                smoothed_signal = prev_signal
            else:
                bars_since_flip = 0
        elif current_direction != 0:
            bars_since_flip += 1
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals