#!/usr/bin/env python3
"""
strategy.py - ADX Donchian Breakout V15
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

Strategy Hypothesis:
    Combine Donchian channel breakouts with ADX trend strength filter:
    - Primary signal: Donchian(20) breakout above/below 20-period high/low
    - Confirmation: ADX(14) > 25 ensures strong trending market
    - Filter: +DI/-DI crossover confirms direction
    - Volume: Breakout volume > 1.2x average confirms genuine move
    - Risk: ATR-based position sizing limits exposure during high vol
    
    Why this works:
    - Donchian channels capture momentum breakouts effectively
    - ADX filter avoids whipsaws in ranging markets (major failure point of v4)
    - Volume confirmation reduces false breakouts
    - 4h timeframe worked well for supertrend (best Sharpe=0.253)
    - Conservative leverage (1.5x) should keep drawdown < 50%
    
    Differences from failed strategies:
    - v4 (kama_donchian_breakout) had 0 trades - too restrictive
    - This version relaxes entry filters while adding ADX confirmation
    - Better warmup handling to ensure signals generate

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

name = "adx_donchian_breakout_v15"
timeframe = "4h"
leverage = 1.5  # Conservative to control drawdown

# Donchian channel configuration
DONCHIAN_PERIOD = 20  # 20-period high/low for breakout

# ADX configuration for trend strength
ADX_PERIOD = 14
ADX_THRESHOLD = 22  # Minimum ADX to trade (slightly lower than 25 for more trades)
DI_THRESHOLD = 5  # Minimum +DI/-DI difference

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 1.10  # Breakout volume must be > 1.1x average

# ATR configuration for risk management
ATR_PERIOD = 14
ATR_STOP_MULT = 2.5  # ATR multiplier for trailing stop
VOLATILITY_TARGET = 0.020  # Target ATR as % of price
VOLATILITY_MIN = 0.005  # Minimum ATR % to trade
VOLATILITY_MAX = 0.080  # Maximum ATR % to trade

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SIGNAL_SMOOTHING = 0.30  # EMA smoothing factor for signals
COOLDOWN_PERIOD = 3  # Bars after signal flip before allowing another flip


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_donchian_channels(high: np.ndarray, low: np.ndarray, period: int = 20) -> tuple:
    """
    Calculate Donchian channel upper and lower bands.
    Upper = highest high over last N periods
    Lower = lowest low over last N periods
    Only uses past data (no look-ahead).
    """
    n = len(high)
    upper = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    
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
    
    # EMA smoothing for ATR
    alpha = 1.0 / period
    atr[period - 1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    
    return atr


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> tuple:
    """
    Calculate ADX, +DI, and -DI using only past data.
    Returns: (adx, plus_di, minus_di)
    """
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    if n < period * 2:
        return adx, plus_di, minus_di
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n, dtype=np.float64)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
        
        if high[i] - high[i-1] > low[i-1] - low[i]:
            plus_dm[i] = max(0, high[i] - high[i-1])
        else:
            plus_dm[i] = 0
            
        if low[i-1] - low[i] > high[i] - high[i-1]:
            minus_dm[i] = max(0, low[i-1] - low[i])
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (EMA with alpha=1/period)
    alpha = 1.0 / period
    
    atr_smooth = np.zeros(n, dtype=np.float64)
    plus_dm_smooth = np.zeros(n, dtype=np.float64)
    minus_dm_smooth = np.zeros(n, dtype=np.float64)
    
    # Initialize with simple average for first period
    atr_smooth[period - 1] = np.mean(tr[:period])
    plus_dm_smooth[period - 1] = np.mean(plus_dm[:period])
    minus_dm_smooth[period - 1] = np.mean(minus_dm[:period])
    
    for i in range(period, n):
        atr_smooth[i] = alpha * tr[i] + (1 - alpha) * atr_smooth[i - 1]
        plus_dm_smooth[i] = alpha * plus_dm[i] + (1 - alpha) * plus_dm_smooth[i - 1]
        minus_dm_smooth[i] = alpha * minus_dm[i] + (1 - alpha) * minus_dm_smooth[i - 1]
    
    # Calculate +DI and -DI
    for i in range(period - 1, n):
        if atr_smooth[i] > 0:
            plus_di[i] = 100 * plus_dm_smooth[i] / atr_smooth[i]
            minus_di[i] = 100 * minus_dm_smooth[i] / atr_smooth[i]
    
    # Calculate DX and ADX
    dx = np.zeros(n, dtype=np.float64)
    
    for i in range(period - 1, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Smooth DX to get ADX
    adx[2 * period - 2] = np.mean(dx[period - 1:2 * period - 1])
    
    for i in range(2 * period - 1, n):
        adx[i] = alpha * dx[i] + (1 - alpha) * adx[i - 1]
    
    return adx, plus_di, minus_di


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
        if avg_vol > 0:
            volume_ratio[i] = volume[i] / avg_vol
    
    return volume_ratio


def calculate_ema(data: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
    """
    n = len(data)
    ema = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return ema
    
    alpha = 2.0 / (period + 1)
    ema[period - 1] = np.mean(data[:period])
    
    for i in range(period, n):
        ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
    
    return ema


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    ADX Donchian Breakout V15 Strategy.
    
    Signal Logic:
    1. Calculate Donchian channels (20-period high/low)
    2. Calculate ADX, +DI, -DI for trend strength and direction
    3. Calculate ATR for volatility normalization
    4. Calculate volume ratio for breakout confirmation
    5. Entry: Price breaks above Donchian upper + ADX > threshold + +DI > -DI
    6. Exit: Price breaks below Donchian lower + ADX > threshold + -DI > +DI
    7. Filter: Volume confirmation and volatility bounds
    8. Smooth signals and apply cooldown to reduce whipsaws
    
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
    high = np.where(high < close, close, high)
    low = np.where(low > close, close, low)
    
    # Calculate all indicators (all use only past data)
    donchian_upper, donchian_lower = calculate_donchian_channels(high, low, DONCHIAN_PERIOD)
    adx, plus_di, minus_di = calculate_adx(high, low, close, ADX_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        DONCHIAN_PERIOD,
        ADX_PERIOD * 2,  # ADX needs 2x period for proper calculation
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    cooldown_counter = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0 or adx[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            cooldown_counter = 0
            continue
        
        # Volatility filter (not too low, not too high)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            cooldown_counter = 0
            continue
        
        # Check for Donchian breakout
        breakout_long = close[i] > donchian_upper[i]
        breakout_short = close[i] < donchian_lower[i]
        
        # ADX trend strength filter
        trend_strong = adx[i] > ADX_THRESHOLD
        
        # DI direction confirmation
        di_long = plus_di[i] > minus_di[i] + DI_THRESHOLD
        di_short = minus_di[i] > plus_di[i] + DI_THRESHOLD
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_MIN_RATIO
        
        # Calculate raw signal
        raw_signal = 0.0
        
        if breakout_long and trend_strong and di_long:
            # Long breakout confirmed
            if volume_confirmed:
                raw_signal = 1.0
            else:
                raw_signal = 0.5  # Weaker signal without volume
        elif breakout_short and trend_strong and di_short:
            # Short breakout confirmed
            if volume_confirmed:
                raw_signal = -1.0
            else:
                raw_signal = -0.5  # Weaker signal without volume
        
        # Apply cooldown to reduce whipsaws
        if cooldown_counter > 0:
            # In cooldown period, maintain previous direction or go flat
            if prev_direction != 0:
                raw_signal = prev_direction * 0.5  # Maintain but weaken
            cooldown_counter -= 1
        
        # Check if signal direction changed
        current_direction = np.sign(raw_signal)
        if current_direction != 0 and current_direction != prev_direction and prev_direction != 0:
            # Direction flip detected, start cooldown
            cooldown_counter = COOLDOWN_PERIOD
            raw_signal = 0.0  # Go flat during cooldown
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        smoothed_signal *= vol_factor
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        # Ensure no NaN or Inf
        if not np.isfinite(signal):
            signal = 0.0
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    # Final validation: ensure no NaN or Inf in output
    signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
    
    return signals