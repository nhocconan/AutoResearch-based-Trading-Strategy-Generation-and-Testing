#!/usr/bin/env python3
"""
strategy.py - Adaptive Regime Trend V3
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on adaptive_regime_trend_v2 success (Sharpe=0.500), improving:
    - Cleaner regime detection logic (simplified from v3/v4 failures)
    - Multi-timeframe trend confirmation (200 EMA as major filter)
    - Enhanced volume breakout confirmation
    - Better regime transition handling
    - Reduced parameter complexity to avoid overfitting
    
    Key improvements over adaptive_regime_trend_v2:
    - Simplified regime confidence calculation
    - Stronger multi-timeframe alignment requirement
    - Volume spike detection for breakout confirmation
    - Cleaner signal combination logic
    - Better volatility-adaptive position sizing

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

name = "adaptive_regime_trend_v3"
timeframe = "1h"
leverage = 2.5  # Conservative leverage for crypto futures

# EMA periods for multi-timeframe trend detection
EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 50
EMA_MAJOR = 200  # Major trend filter (daily equivalent on 1h)

# RSI configuration
RSI_PERIOD = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
RSI_EXTREME_HIGH = 75
RSI_EXTREME_LOW = 25

# ADX regime detection
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_STRONG_THRESHOLD = 35
ADX_WEAK_THRESHOLD = 20

# Bollinger Band configuration
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_THRESHOLD = 0.02  # BB width below this = squeeze

# MACD configuration
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_THRESHOLD = 1.5  # Volume > 1.5x average = spike

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012
VOLATILITY_MIN = 0.003
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL_TRENDING = 0.15
MIN_SIGNAL_RANGING = 0.25
MIN_SIGNAL_BREAKOUT = 0.30
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.65
HYSTERESIS_THRESHOLD = 0.08

# Regime transition smoothing
REGIME_MEMORY = 3  # Bars to remember previous regime


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


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average Directional Index using only past data.
    ADX measures trend strength (not direction).
    """
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    
    if n < period * 2 + 1:
        return adx
    
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
        
        plus_dm[i] = max(high[i] - high[i-1], 0.0)
        minus_dm[i] = max(low[i-1] - low[i], 0.0)
        
        # True DM rules
        if plus_dm[i] > minus_dm[i]:
            minus_dm[i] = 0.0
        elif minus_dm[i] > plus_dm[i]:
            plus_dm[i] = 0.0
    
    # Smooth with EMA
    tr_series = pd.Series(tr)
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    
    atr_series = tr_series.ewm(span=period, adjust=False, min_periods=period).mean()
    plus_di_series = (plus_dm_series.ewm(span=period, adjust=False, min_periods=period).mean() / 
                      atr_series * 100)
    minus_di_series = (minus_dm_series.ewm(span=period, adjust=False, min_periods=period).mean() / 
                       atr_series * 100)
    
    # Calculate DX and ADX
    dx = 100 * np.abs(plus_di_series - minus_di_series) / (plus_di_series + minus_di_series).replace(0, np.inf)
    adx_series = dx.ewm(span=period, adjust=False, min_periods=period).mean()
    
    adx = np.nan_to_num(adx_series.values, nan=0.0)
    
    return adx


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands using only past data.
    Returns: (upper, middle, lower, bandwidth)
    """
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    bandwidth = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower, bandwidth
    
    close_series = pd.Series(close)
    middle_series = close_series.rolling(window=period, min_periods=period).mean()
    std_series = close_series.rolling(window=period, min_periods=period).std()
    
    upper = np.nan_to_num((middle_series + std_dev * std_series).values, nan=0.0)
    middle = np.nan_to_num(middle_series.values, nan=0.0)
    lower = np.nan_to_num((middle_series - std_dev * std_series).values, nan=0.0)
    bandwidth = np.where(middle > 0, (upper - lower) / middle, 0.0)
    
    return upper, middle, lower, bandwidth


def calculate_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """
    Calculate MACD indicator using only past data.
    Returns: (macd_line, signal_line, histogram)
    """
    n = len(close)
    macd_line = np.zeros(n, dtype=np.float64)
    signal_line = np.zeros(n, dtype=np.float64)
    histogram = np.zeros(n, dtype=np.float64)
    
    if n < slow + signal:
        return macd_line, signal_line, histogram
    
    close_series = pd.Series(close)
    ema_fast = close_series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    
    macd_series = ema_fast - ema_slow
    signal_series = macd_series.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist_series = macd_series - signal_series
    
    macd_line = np.nan_to_num(macd_series.values, nan=0.0)
    signal_line = np.nan_to_num(signal_series.values, nan=0.0)
    histogram = np.nan_to_num(hist_series.values, nan=0.0)
    
    return macd_line, signal_line, histogram


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio vs rolling average using only past data.
    Returns ratio where >1.0 means above average volume.
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=lookback, min_periods=lookback).mean()
    
    volume_ratio = np.nan_to_num((volume / avg_volume.values), nan=1.0)
    
    return volume_ratio


def calculate_ema_alignment(ema_fast: float, ema_medium: float, 
                            ema_slow: float, ema_major: float) -> float:
    """
    Calculate EMA stack alignment score.
    Returns value in [-1, 1] where:
    - Positive = bullish alignment (fast > medium > slow > major)
    - Negative = bearish alignment (fast < medium < slow < major)
    - Magnitude = strength of alignment
    """
    # Check bullish alignment
    bullish = (ema_fast > ema_medium > ema_slow > ema_major)
    bearish = (ema_fast < ema_medium < ema_slow < ema_major)
    
    if bullish:
        # Calculate alignment strength
        spread1 = (ema_fast - ema_medium) / ema_major
        spread2 = (ema_medium - ema_slow) / ema_major
        spread3 = (ema_slow - ema_major) / ema_major
        strength = (spread1 + spread2 + spread3) / 3
        return np.clip(strength * 50, 0.1, 1.0)
    elif bearish:
        # Calculate alignment strength
        spread1 = (ema_major - ema_slow) / ema_major
        spread2 = (ema_slow - ema_medium) / ema_major
        spread3 = (ema_medium - ema_fast) / ema_major
        strength = (spread1 + spread2 + spread3) / 3
        return np.clip(-strength * 50, -1.0, -0.1)
    else:
        # Mixed alignment - calculate net bias
        bias = 0.0
        if ema_fast > ema_medium:
            bias += 0.25
        else:
            bias -= 0.25
        if ema_medium > ema_slow:
            bias += 0.35
        else:
            bias -= 0.35
        if ema_slow > ema_major:
            bias += 0.4
        else:
            bias -= 0.4
        return np.clip(bias, -1.0, 1.0)


def determine_regime(adx: float, bb_width: float, volume_ratio: float) -> str:
    """
    Determine market regime based on ADX, BB width, and volume.
    Returns: 'trending', 'ranging', 'breakout', or 'transition'
    """
    is_squeeze = bb_width < BB_SQUEEZE_THRESHOLD
    is_high_volume = volume_ratio > VOLUME_SPIKE_THRESHOLD
    
    if is_squeeze and is_high_volume:
        return 'breakout'
    elif adx >= ADX_TREND_THRESHOLD:
        return 'trending'
    elif adx <= ADX_WEAK_THRESHOLD:
        return 'ranging'
    else:
        return 'transition'


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Regime Trend V3 Strategy.
    
    Signal Logic:
    1. Multi-timeframe trend filter (200 EMA as major direction)
    2. Regime detection (trending/ranging/breakout/transition)
    3. EMA alignment confirmation for trend direction
    4. RSI for entry timing and overbought/oversold detection
    5. Volume confirmation for breakouts
    6. Volatility-adaptive position sizing
    7. Signal smoothing with hysteresis
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract required columns with safety checks
    try:
        close = prices["close"].values.astype(np.float64)
        high = prices["high"].values.astype(np.float64)
        low = prices["low"].values.astype(np.float64)
        volume = prices["volume"].values.astype(np.float64)
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Handle NaN values
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Ensure valid prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_medium = calculate_ema(close, EMA_MEDIUM)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    macd_line, macd_signal, macd_hist = calculate_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR + 10,  # Extra buffer for major EMA stability
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        ADX_PERIOD * 2 + 1,
        VOLUME_LOOKBACK,
        BB_PERIOD,
        MACD_SLOW + MACD_SIGNAL
    )
    
    # Track previous signal for smoothing and hysteresis
    prev_signal = 0.0
    prev_direction = 0  # 0=neutral, 1=long, -1=short
    regime_memory = [0] * REGIME_MEMORY  # Track recent regimes (1=trending, 0=ranging)
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0 or ema_major[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Check volatility regime
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Determine market regime
        regime = determine_regime(adx[i], bb_width[i], volume_ratio[i])
        
        # Update regime memory
        regime_memory.pop(0)
        regime_memory.append(1 if regime == 'trending' else 0)
        recent_trending_ratio = sum(regime_memory) / REGIME_MEMORY
        
        # Calculate EMA alignment score
        ema_alignment = calculate_ema_alignment(
            ema_fast[i], ema_medium[i], ema_slow[i], ema_major[i]
        )
        
        # Determine major trend direction from 200 EMA
        major_trend = np.sign(close[i] - ema_major[i])
        
        # Initialize raw signal
        raw_signal = 0.0
        regime_weight = 0.0
        
        # BREAKOUT REGIME (squeeze + volume spike)
        if regime == 'breakout':
            regime_weight = MIN_SIGNAL_BREAKOUT
            
            # Breakout direction from EMA alignment and price position
            if ema_alignment > 0.2 and major_trend > 0:
                # Bullish breakout
                breakout_strength = min(volume_ratio[i] / VOLUME_SPIKE_THRESHOLD, 2.0)
                raw_signal = 0.5 * breakout_strength * ema_alignment
            elif ema_alignment < -0.2 and major_trend < 0:
                # Bearish breakout
                breakout_strength = min(volume_ratio[i] / VOLUME_SPIKE_THRESHOLD, 2.0)
                raw_signal = -0.5 * breakout_strength * abs(ema_alignment)
            else:
                # Unclear breakout direction
                raw_signal = 0.0
        
        # TRENDING REGIME
        elif regime == 'trending':
            regime_weight = MIN_SIGNAL_TRENDING
            
            # Trade with major trend direction
            if major_trend > 0:
                # Long bias - only take long or neutral
                if ema_alignment > 0.1:
                    raw_signal = ema_alignment
                    # Amplify in strong trends
                    if adx[i] >= ADX_STRONG_THRESHOLD:
                        raw_signal *= 1.15
                else:
                    # EMA not aligned but price above major - wait
                    raw_signal = 0.0
            else:
                # Short bias - only take short or neutral
                if ema_alignment < -0.1:
                    raw_signal = ema_alignment
                    # Amplify in strong trends
                    if adx[i] >= ADX_STRONG_THRESHOLD:
                        raw_signal *= 1.15
                else:
                    # EMA not aligned but price below major - wait
                    raw_signal = 0.0
            
            # RSI filter - avoid entering at extremes against trend
            if major_trend > 0 and rsi[i] > RSI_EXTREME_HIGH:
                raw_signal *= 0.5  # Reduce long at overbought
            elif major_trend < 0 and rsi[i] < RSI_EXTREME_LOW:
                raw_signal *= 0.5  # Reduce short at oversold
        
        # RANGING REGIME
        elif regime == 'ranging':
            regime_weight = MIN_SIGNAL_RANGING
            
            # Mean reversion based on RSI
            if rsi[i] < RSI_OVERSOLD:
                raw_signal = 0.35  # Long signal
                # Amplify at extreme oversold
                if rsi[i] < RSI_EXTREME_LOW:
                    raw_signal = 0.5
            elif rsi[i] > RSI_OVERBOUGHT:
                raw_signal = -0.35  # Short signal
                # Amplify at extreme overbought
                if rsi[i] > RSI_EXTREME_HIGH:
                    raw_signal = -0.5
            else:
                raw_signal = 0.0
            
            # Filter against major trend (fade moves against trend)
            if major_trend > 0:
                # In uptrend, prefer long mean reversion, avoid short
                if raw_signal < 0:
                    raw_signal *= 0.3
            else:
                # In downtrend, prefer short mean reversion, avoid long
                if raw_signal > 0:
                    raw_signal *= 0.3
        
        # TRANSITION REGIME (mixed signals)
        else:  # regime == 'transition'
            regime_weight = MIN_SIGNAL_RANGING * 0.7
            
            # Blend trend and mean reversion based on recent regime history
            trend_signal = ema_alignment * recent_trending_ratio
            rsi_signal = 0.0
            
            if rsi[i] < RSI_OVERSOLD:
                rsi_signal = 0.25
            elif rsi[i] > RSI_OVERBOUGHT:
                rsi_signal = -0.25
            
            # Weight by recent regime tendency
            raw_signal = recent_trending_ratio * trend_signal + (1 - recent_trending_ratio) * rsi_signal
        
        # MACD confirmation (reduce signal if MACD disagrees)
        macd_conf = 1.0
        if raw_signal > 0 and macd_hist[i] < 0:
            macd_conf = 0.7
        elif raw_signal < 0 and macd_hist[i] > 0:
            macd_conf = 0.7
        raw_signal *= macd_conf
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        
        raw_signal *= vol_factor
        
        # Apply exponential smoothing
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Apply hysteresis to reduce flipping
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            # Check if signal change exceeds hysteresis threshold
            if abs(smoothed_signal - prev_signal) < HYSTERESIS_THRESHOLD:
                smoothed_signal = prev_signal  # Keep previous direction
        
        # Apply minimum signal threshold based on regime
        if abs(smoothed_signal) < regime_weight:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals