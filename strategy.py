#!/usr/bin/env python3
"""
strategy.py - Multi-TF Trend Volatility V15
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "4h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    4h timeframe provides cleaner trend signals than 1h with fewer whipsaws.
    Combine:
    - Primary: EMA trend direction (34/89 EMA crossover)
    - Filter: Price above/below 200 EMA for major trend
    - Momentum: RSI confirmation (not extreme)
    - Volatility: ATR-based position sizing to control drawdown
    - Funding: Contrarian overlay on extreme funding rates
    
    Key improvements over v12:
    - 4h timeframe for cleaner signals (less noise)
    - Volatility regime detection to reduce exposure in high vol
    - Simpler signal logic for better robustness
    - Conservative leverage (1.5x) to control drawdown
    - Better position sizing based on ATR

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

name = "multi_tf_trend_vol_v15"
timeframe = "4h"
leverage = 1.5  # Conservative leverage to control drawdown

# EMA configuration for trend detection
EMA_FAST = 34
EMA_SLOW = 89
EMA_MAJOR = 200

# RSI configuration for momentum
RSI_PERIOD = 14
RSI_LONG_MIN = 45  # Minimum RSI for long entries
RSI_SHORT_MAX = 55  # Maximum RSI for short entries
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Funding rate configuration
FUNDING_EXTREME = 0.0015  # 0.15% per 8hr = extreme
FUNDING_MODERATE = 0.0005  # 0.05% per 8hr = moderate
FUNDING_LOOKBACK = 80
FUNDING_WEIGHT = 0.25  # Funding influence on signal

# Volatility configuration
ATR_PERIOD = 14
VOL_TARGET = 0.02  # Target ATR as % of price
VOL_LOW = 0.005  # Minimum ATR % to trade
VOL_HIGH = 0.08  # Maximum ATR % (reduce exposure above this)
VOL_REGIME_LOOKBACK = 50  # For volatility regime detection

# Signal configuration
MIN_SIGNAL = 0.20  # Minimum signal magnitude to trade
MAX_SIGNAL = 0.75  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.4  # EMA smoothing factor for signals
DIRECTION_CHANGE_MIN = 0.15  # Minimum change to flip direction

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.6


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA using only past data."""
    n = len(close)
    if n < period:
        return np.zeros(n, dtype=np.float64)
    
    close_series = pd.Series(close)
    ema = close_series.ewm(span=period, adjust=False, min_periods=period).mean()
    return np.nan_to_num(ema.values, nan=0.0)


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI using only past data."""
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
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return np.nan_to_num(rsi.values, nan=50.0)


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR using only past data."""
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
    atr = tr_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    return np.nan_to_num(atr.values, nan=0.0)


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Calculate volume ratio vs rolling average (past data only)."""
    n = len(volume)
    if n < lookback:
        return np.ones(n, dtype=np.float64)
    
    volume_series = pd.Series(volume)
    rolling_avg = volume_series.rolling(window=lookback, min_periods=lookback).mean()
    
    return np.nan_to_num(volume_series.values / rolling_avg.values, nan=1.0)


def calculate_volatility_regime(atr_pct: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calculate volatility regime: 0=low, 1=normal, 2=high.
    Based on percentile of recent ATR values.
    Uses only past data.
    """
    n = len(atr_pct)
    regime = np.ones(n, dtype=np.float64)  # Default: normal
    
    if n < lookback:
        return regime
    
    for i in range(lookback, n):
        recent_vol = atr_pct[i-lookback:i+1]
        percentile = np.percentile(recent_vol, 70)
        
        if atr_pct[i] < percentile * 0.5:
            regime[i] = 0.0  # Low vol
        elif atr_pct[i] > percentile * 1.5:
            regime[i] = 2.0  # High vol
        else:
            regime[i] = 1.0  # Normal
    
    return regime


def calculate_funding_signal(funding_rate: np.ndarray, lookback: int = 80) -> np.ndarray:
    """
    Calculate funding rate contrarian signal.
    Extreme positive → short bias, extreme negative → long bias.
    Uses only past data.
    """
    n = len(funding_rate)
    signal = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return signal
    
    for i in range(lookback, n):
        fr = funding_rate[i]
        
        # Calculate recent funding extremes
        recent_funding = funding_rate[i-lookback:i+1]
        high_percentile = np.percentile(recent_funding, 90)
        low_percentile = np.percentile(recent_funding, 10)
        
        # Determine signal based on funding level
        if fr > FUNDING_EXTREME or fr >= high_percentile * 0.9:
            # Extreme positive funding → short bias
            signal[i] = -FUNDING_WEIGHT * min(1.0, fr / FUNDING_EXTREME)
        elif fr < -FUNDING_EXTREME or fr <= low_percentile * 0.9:
            # Extreme negative funding → long bias
            signal[i] = FUNDING_WEIGHT * min(1.0, abs(fr) / FUNDING_EXTREME)
        elif fr > FUNDING_MODERATE:
            # Moderate positive → mild short bias
            signal[i] = -FUNDING_WEIGHT * 0.3 * (fr / FUNDING_MODERATE)
        elif fr < -FUNDING_MODERATE:
            # Moderate negative → mild long bias
            signal[i] = FUNDING_WEIGHT * 0.3 * (abs(fr) / FUNDING_MODERATE)
    
    return signal


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Multi-TF Trend Volatility V15 Strategy.
    
    Signal Logic:
    1. Calculate EMA trend (34/89 crossover with 200 EMA filter)
    2. Confirm with RSI momentum
    3. Adjust for volatility regime (reduce exposure in high vol)
    4. Add funding rate contrarian overlay
    5. Smooth signals and apply hysteresis
    6. Filter by minimum magnitude
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, funding_rate]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract and clean price data
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
    
    # Clean invalid data
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
    funding_signal = calculate_funding_signal(funding_rate, FUNDING_LOOKBACK)
    
    # Calculate ATR as % of price for volatility metrics
    atr_pct = np.zeros(n, dtype=np.float64)
    valid_mask = close > 0
    atr_pct[valid_mask] = atr[valid_mask] / close[valid_mask]
    
    # Calculate volatility regime
    vol_regime = calculate_volatility_regime(atr_pct, VOL_REGIME_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW + 10,  # Extra buffer for stability
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
        
        # Volatility filter
        if atr_pct[i] < VOL_LOW or atr_pct[i] > VOL_HIGH:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volume filter
        if volume_ratio[i] < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # === Trend Signal ===
        ema_diff = (ema_fast[i] - ema_slow[i]) / close[i]
        ema_direction = np.sign(ema_diff)
        
        # Major trend filter
        major_filter = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend
        if ema_direction != major_filter and abs(ema_direction) > 0:
            trend_strength = abs(ema_diff) * 40 * 0.5  # Reduce on conflict
        else:
            trend_strength = abs(ema_diff) * 40
        
        # === RSI Momentum Filter ===
        rsi_factor = 1.0
        if ema_direction > 0:
            if rsi[i] < RSI_LONG_MIN:
                rsi_factor = 0.0
            elif rsi[i] > RSI_OVERBOUGHT:
                rsi_factor = 0.5
        elif ema_direction < 0:
            if rsi[i] > RSI_SHORT_MAX:
                rsi_factor = 0.0
            elif rsi[i] < RSI_OVERSOLD:
                rsi_factor = 0.5
        
        trend_signal = ema_direction * trend_strength * rsi_factor
        
        # === Volatility Regime Adjustment ===
        vol_multiplier = 1.0
        if vol_regime[i] == 2.0:  # High volatility
            vol_multiplier = 0.5  # Reduce exposure
        elif vol_regime[i] == 0.0:  # Low volatility
            vol_multiplier = 1.2  # Slightly increase
        
        # === Combine Signals ===
        fund_sig = funding_signal[i]
        
        # Combine trend and funding
        if abs(trend_signal) > 0.2 and abs(fund_sig) > 0.05:
            if np.sign(trend_signal) != np.sign(fund_sig):
                # Conflict: reduce trend strength
                raw_signal = trend_signal * 0.7 + fund_sig
            else:
                # Aligned: reinforce
                raw_signal = trend_signal * 0.75 + fund_sig * 0.25
        else:
            raw_signal = trend_signal * 0.8 + fund_sig * 0.2
        
        # Apply volatility regime multiplier
        raw_signal *= vol_multiplier
        
        # === Signal Smoothing ===
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # === Hysteresis ===
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < DIRECTION_CHANGE_MIN:
                smoothed_signal = prev_signal
        
        # === Minimum Magnitude Filter ===
        if abs(smoothed_signal) < MIN_SIGNAL:
            smoothed_signal = 0.0
        
        # === Clip to Max ===
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals