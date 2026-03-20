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
    - Enhanced regime confidence scoring with multi-factor validation
    - Improved EMA stack alignment with dynamic weighting
    - Better momentum confirmation using RSI + MACD confluence
    - More robust volatility-based position sizing
    - Smoother regime transitions with exponential memory decay
    - Better handling of false breakouts from BB squeezes
    
    Key improvements over adaptive_regime_trend_v2:
    - Multi-factor regime confidence (ADX + BB + ATR + Volume)
    - Dynamic EMA weighting based on volatility regime
    - Momentum confluence score (RSI + MACD + Price action)
    - Improved signal smoothing with adaptive hysteresis
    - Better volume confirmation for breakouts
    - Reduced parameter sensitivity

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

# EMA periods for trend detection (optimized for 1h timeframe)
EMA_FAST = 8
EMA_MEDIUM = 21
EMA_SLOW = 55
EMA_MAJOR = 200

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
BB_SQUEEZE_THRESHOLD = 0.012  # Slightly tighter for better breakout detection

# MACD configuration
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_THRESHOLD = 1.5  # Volume > 1.5x average = spike

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.008  # Target ATR as % of price
VOLATILITY_MIN = 0.0015
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL_TRENDING = 0.15
MIN_SIGNAL_RANGING = 0.18
MIN_SIGNAL_BREAKOUT = 0.22
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.65  # Slightly less smoothing for faster response
HYSTERESIS_THRESHOLD = 0.08

# Regime transition smoothing
REGIME_MEMORY_DECAY = 0.7  # Exponential decay factor for regime memory


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
    Calculate volume ratio (current / rolling average) using only past data.
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=lookback, min_periods=lookback).mean()
    
    volume_ratio = np.nan_to_num((volume_series / avg_volume).values, nan=1.0)
    
    return volume_ratio


def calculate_momentum_confluence(rsi: np.ndarray, macd_hist: np.ndarray, 
                                   close: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Calculate combined momentum score from multiple indicators.
    Returns value in [-1, 1].
    """
    n = len(close)
    momentum = np.zeros(n, dtype=np.float64)
    
    if n < lookback + 1:
        return momentum
    
    for i in range(lookback + 1, n):
        # RSI momentum (slope and level)
        rsi_slope = (rsi[i] - rsi[i-lookback]) / 50.0  # Normalize
        rsi_level = (rsi[i] - 50) / 50.0  # Centered around 50
        
        # MACD histogram momentum
        macd_slope = (macd_hist[i] - macd_hist[i-lookback]) / (close[i] * 0.01) if close[i] > 0 else 0
        macd_level = np.clip(macd_hist[i] / (close[i] * 0.01), -1, 1) if close[i] > 0 else 0
        
        # Price momentum
        price_momentum = (close[i] - close[i-lookback]) / close[i-lookback] if close[i-lookback] > 0 else 0
        price_momentum = np.clip(price_momentum * 10, -1, 1)
        
        # Combine with weights
        momentum[i] = (
            0.3 * np.clip(rsi_slope, -1, 1) +
            0.2 * np.clip(rsi_level, -1, 1) +
            0.2 * np.clip(macd_slope, -1, 1) +
            0.15 * np.clip(macd_level, -1, 1) +
            0.15 * price_momentum
        )
    
    return np.clip(momentum, -1, 1)


def calculate_ema_stack_alignment(ema_fast: float, ema_medium: float, 
                                   ema_slow: float, ema_major: float,
                                   close: float) -> tuple:
    """
    Calculate EMA stack alignment score and direction.
    Returns: (alignment_score, direction)
    alignment_score in [0, 1], direction in [-1, 0, 1]
    """
    if close <= 0 or ema_major <= 0:
        return 0.0, 0
    
    # Check bullish alignment: close > fast > medium > slow > major
    bullish = (close > ema_fast > ema_medium > ema_slow > ema_major)
    
    # Check bearish alignment: close < fast < medium < slow < major
    bearish = (close < ema_fast < ema_medium < ema_slow < ema_major)
    
    # Calculate alignment score based on spacing
    if bullish:
        direction = 1
        spacing1 = (ema_fast - ema_medium) / close
        spacing2 = (ema_medium - ema_slow) / close
        spacing3 = (ema_slow - ema_major) / close
        alignment_score = min(1.0, (abs(spacing1) + abs(spacing2) + abs(spacing3)) * 50)
    elif bearish:
        direction = -1
        spacing1 = (ema_medium - ema_fast) / close
        spacing2 = (ema_slow - ema_medium) / close
        spacing3 = (ema_major - ema_slow) / close
        alignment_score = min(1.0, (abs(spacing1) + abs(spacing2) + abs(spacing3)) * 50)
    else:
        # Mixed alignment
        direction = 0
        # Calculate partial alignment score
        score = 0.0
        if ema_fast > ema_medium:
            score += 0.25
        else:
            score -= 0.25
        if ema_medium > ema_slow:
            score += 0.25
        else:
            score -= 0.25
        if ema_slow > ema_major:
            score += 0.25
        else:
            score -= 0.25
        if close > ema_major:
            score += 0.25
        else:
            score -= 0.25
        alignment_score = abs(score)
        direction = np.sign(score)
    
    return np.clip(alignment_score, 0, 1), direction


def calculate_regime_confidence_multi(adx: float, bb_width: float, atr_pct: float,
                                       volume_ratio: float, adx_trend: float,
                                       adx_weak: float, bb_squeeze: float) -> dict:
    """
    Calculate multi-factor regime confidence scores.
    Returns dict with confidence scores for each regime.
    """
    # ADX-based trend confidence
    if adx >= adx_trend:
        adx_conf = np.clip((adx - adx_trend) / 25 + 0.5, 0.5, 1.0)
    elif adx >= adx_weak:
        adx_conf = np.clip((adx - adx_weak) / (adx_trend - adx_weak) * 0.5, 0.2, 0.5)
    else:
        adx_conf = np.clip(adx / adx_weak * 0.2, 0.0, 0.2)
    
    # BB width-based ranging confidence
    if bb_width < bb_squeeze:
        bb_ranging_conf = 0.3  # Squeeze = potential breakout, not stable ranging
    elif bb_width < bb_squeeze * 2:
        bb_ranging_conf = 0.6
    else:
        bb_ranging_conf = 0.8
    
    # ATR-based volatility confidence
    if atr_pct < 0.003:
        vol_conf = 0.3  # Very low vol = potential breakout
    elif atr_pct < 0.015:
        vol_conf = 0.7  # Normal vol
    else:
        vol_conf = 0.4  # High vol = unstable
    
    # Volume confirmation
    vol_conf_factor = min(1.0, volume_ratio / 1.5)
    
    # Calculate regime confidences
    trending_conf = adx_conf * vol_conf_factor
    ranging_conf = bb_ranging_conf * (1 - adx_conf) * vol_conf
    breakout_conf = 0.0
    
    if bb_width < bb_squeeze:
        breakout_conf = 0.5 + 0.5 * (1 - bb_width / bb_squeeze)
        breakout_conf *= (1 - ranging_conf)  # Reduce if ranging is strong
    
    # Normalize
    total = trending_conf + ranging_conf + breakout_conf
    if total > 0:
        trending_conf /= total
        ranging_conf /= total
        breakout_conf /= total
    
    return {
        'trending': trending_conf,
        'ranging': ranging_conf,
        'breakout': breakout_conf
    }


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Regime Trend V3 Strategy.
    
    Signal Logic:
    1. Calculate multi-factor regime confidence
    2. Apply regime-weighted signal logic
    3. EMA stack alignment for trend confirmation
    4. Momentum confluence from RSI + MACD + Price
    5. Volume confirmation for breakouts
    6. Volatility-adaptive position sizing
    7. Signal smoothing with adaptive hysteresis
    8. Regime memory with exponential decay
    
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
    momentum_conf = calculate_momentum_confluence(rsi, macd_hist, close, lookback=5)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        ADX_PERIOD * 2 + 1,
        VOLUME_LOOKBACK,
        BB_PERIOD,
        MACD_SLOW + MACD_SIGNAL
    )
    
    # Track state for smoothing and memory
    prev_signal = 0.0
    prev_direction = 0  # 0=neutral, 1=long, -1=short
    regime_memory = {'trending': 0.5, 'ranging': 0.3, 'breakout': 0.2}  # Exponential memory
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0:
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
        
        # Calculate multi-factor regime confidence
        regime_conf = calculate_regime_confidence_multi(
            adx[i], bb_width[i], atr_pct, volume_ratio[i],
            ADX_TREND_THRESHOLD, ADX_WEAK_THRESHOLD, BB_SQUEEZE_THRESHOLD
        )
        
        # Update regime memory with exponential decay
        for key in regime_memory:
            regime_memory[key] = (
                REGIME_MEMORY_DECAY * regime_memory[key] +
                (1 - REGIME_MEMORY_DECAY) * regime_conf[key]
            )
        
        # Calculate EMA stack alignment
        alignment_score, alignment_dir = calculate_ema_stack_alignment(
            ema_fast[i], ema_medium[i], ema_slow[i], ema_major[i], close[i]
        )
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_SPIKE_THRESHOLD
        
        # Initialize raw signal
        raw_signal = 0.0
        regime_weight = 0.0
        
        # Determine dominant regime from memory (smoother than instant)
        dominant_regime = max(regime_memory, key=regime_memory.get)
        
        # BREAKOUT REGIME
        if dominant_regime == 'breakout' and regime_memory['breakout'] > 0.35:
            regime_weight = MIN_SIGNAL_BREAKOUT
            
            # Wait for directional confirmation with volume
            if alignment_dir != 0 and alignment_score > 0.3:
                if alignment_dir > 0 and volume_confirmed:
                    raw_signal = alignment_score * (0.5 + regime_memory['breakout'] * 0.5)
                elif alignment_dir < 0 and volume_confirmed:
                    raw_signal = -alignment_score * (0.5 + regime_memory['breakout'] * 0.5)
                else:
                    # No volume confirmation = reduce signal
                    raw_signal = alignment_dir * alignment_score * 0.3
            
            # Momentum confirmation for breakout
            if abs(momentum_conf[i]) > 0.2:
                raw_signal *= (1 + 0.3 * np.sign(raw_signal) * np.sign(momentum_conf[i]))
        
        # TRENDING REGIME
        elif dominant_regime == 'trending' and regime_memory['trending'] > 0.35:
            regime_weight = MIN_SIGNAL_TRENDING
            
            # Base signal from alignment
            raw_signal = alignment_dir * alignment_score
            
            # Amplify in strong trends
            if adx[i] >= ADX_STRONG_THRESHOLD:
                raw_signal *= 1.15
            
            # Momentum confluence
            if abs(momentum_conf[i]) > 0.15:
                momentum_conf_factor = 0.7 + 0.3 * np.sign(raw_signal) * np.sign(momentum_conf[i])
                raw_signal *= momentum_conf_factor
            
            # Volume boost
            if volume_confirmed:
                raw_signal *= 1.1
            
            # RSI filter (avoid extreme overbought/oversold in trends)
            if raw_signal > 0 and rsi[i] > RSI_EXTREME_HIGH:
                raw_signal *= 0.7
            elif raw_signal < 0 and rsi[i] < RSI_EXTREME_LOW:
                raw_signal *= 0.7
        
        # RANGING REGIME
        elif dominant_regime == 'ranging' and regime_memory['ranging'] > 0.35:
            regime_weight = MIN_SIGNAL_RANGING
            
            # RSI-based mean reversion
            if rsi[i] < RSI_OVERSOLD:
                raw_signal = 0.35 + 0.15 * (RSI_OVERSOLD - rsi[i]) / 20
            elif rsi[i] > RSI_OVERBOUGHT:
                raw_signal = -0.35 - 0.15 * (rsi[i] - RSI_OVERBOUGHT) / 20
            else:
                raw_signal = 0.0
            
            # Amplify at extremes
            if rsi[i] < RSI_EXTREME_LOW:
                raw_signal = min(raw_signal, -0.5) if raw_signal < 0 else max(raw_signal, 0.5)
            elif rsi[i] > RSI_EXTREME_HIGH:
                raw_signal = max(raw_signal, 0.5) if raw_signal > 0 else min(raw_signal, -0.5)
            
            # Reduce if alignment fights mean reversion
            if alignment_dir != 0 and np.sign(raw_signal) != alignment_dir:
                raw_signal *= 0.5
        
        # MIXED REGIME (no dominant regime)
        else:
            regime_weight = MIN_SIGNAL_RANGING * 0.7
            
            # Weighted combination of trend and mean reversion
            trend_component = alignment_dir * alignment_score * regime_memory['trending']
            rsi_component = 0.0
            if rsi[i] < RSI_OVERSOLD:
                rsi_component = 0.3 * regime_memory['ranging']
            elif rsi[i] > RSI_OVERBOUGHT:
                rsi_component = -0.3 * regime_memory['ranging']
            
            raw_signal = trend_component + rsi_component
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        
        raw_signal *= vol_factor
        
        # Apply exponential smoothing
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Apply adaptive hysteresis to reduce flipping
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