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
    - Simplified regime detection (less overfitting risk)
    - Better signal smoothing to reduce whipsaws
    - More robust volatility-based position sizing
    - Cleaner trend confirmation with EMA stack
    - Improved handling of regime transitions
    - Volume spike detection for breakout confirmation
    
    Key improvements over adaptive_regime_trend_v2:
    - Reduced parameter complexity (fewer thresholds to tune)
    - Better signal persistence (reduce flip-flopping)
    - More conservative in uncertain regimes
    - Cleaner separation of trending vs ranging logic
    - Improved volatility normalization

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
leverage = 2.0  # Conservative leverage for better risk-adjusted returns

# EMA periods for trend detection (simplified from v2)
EMA_FAST = 12
EMA_MEDIUM = 26
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration (simplified thresholds)
RSI_PERIOD = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
RSI_EXTREME_HIGH = 75
RSI_EXTREME_LOW = 25

# ADX regime detection (simplified)
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_WEAK_THRESHOLD = 20

# Bollinger Band configuration
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_THRESHOLD = 0.012  # Slightly tighter than v2

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_THRESHOLD = 1.5  # Volume > 1.5x average = spike

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.008  # Slightly lower target for conservative sizing
VOLATILITY_MIN = 0.001
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL_TRENDING = 0.15
MIN_SIGNAL_RANGING = 0.18
MAX_SIGNAL = 0.75  # Slightly reduced from v2
SMOOTHING_FACTOR = 0.75  # More smoothing than v2 (0.7)
HYSTERESIS_THRESHOLD = 0.08  # Slightly higher to reduce flips

# Regime transition smoothing
REGIME_MEMORY = 7  # More bars for regime memory (was 5)


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


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio vs rolling average using only past data.
    Returns ratio (1.0 = average volume).
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=lookback, min_periods=lookback).mean()
    
    volume_ratio = np.nan_to_num(volume_series.values / avg_volume.values, nan=1.0)
    
    return volume_ratio


def calculate_ema_stack_score(ema_fast: float, ema_medium: float, 
                               ema_slow: float, ema_major: float,
                               close: float) -> float:
    """
    Calculate trend direction and strength from EMA stack alignment.
    Returns value in [-1, 1] where sign = direction, magnitude = strength.
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    # Determine major trend direction
    major_trend = np.sign(close - ema_major)
    
    if major_trend == 0:
        return 0.0
    
    # Check EMA stack alignment
    if major_trend > 0:
        # Bullish: expect fast > medium > slow > major
        alignment_score = 0.0
        if ema_fast > ema_medium:
            alignment_score += 0.35
        if ema_medium > ema_slow:
            alignment_score += 0.35
        if ema_slow > ema_major:
            alignment_score += 0.30
        trend_score = alignment_score
    else:
        # Bearish: expect fast < medium < slow < major
        alignment_score = 0.0
        if ema_fast < ema_medium:
            alignment_score += 0.35
        if ema_medium < ema_slow:
            alignment_score += 0.35
        if ema_slow < ema_major:
            alignment_score += 0.30
        trend_score = -alignment_score
    
    # Scale by price deviation from major EMA
    deviation = abs(close - ema_major) / ema_major
    deviation_factor = np.clip(deviation * 50, 0.5, 2.0)
    
    return np.clip(trend_score * deviation_factor * major_trend, -1.0, 1.0)


def determine_regime(adx: float, bb_width: float, adx_trend: float, 
                     adx_weak: float, bb_squeeze: float) -> str:
    """
    Determine market regime based on ADX and Bollinger Band width.
    Returns: 'trending', 'ranging', 'breakout', or 'transition'
    """
    is_squeeze = bb_width < bb_squeeze
    is_strong_trend = adx >= adx_trend
    is_weak = adx < adx_weak
    
    if is_squeeze:
        if is_strong_trend:
            return 'breakout'  # Squeeze with strong trend = potential breakout
        else:
            return 'transition'  # Squeeze without clear trend
    elif is_strong_trend:
        return 'trending'
    elif is_weak:
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
    1. Calculate regime (trending/ranging/breakout/transition)
    2. Apply regime-specific signal logic
    3. Volume confirmation for breakouts
    4. EMA stack for trend direction and strength
    5. RSI for mean reversion in ranging markets
    6. Volatility-adaptive position sizing
    7. Signal smoothing with hysteresis and regime memory
    
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
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR + 10,  # Extra buffer for major EMA stabilization
        RSI_PERIOD + 5,
        ATR_PERIOD + 5,
        ADX_PERIOD * 2 + 5,
        VOLUME_LOOKBACK + 5,
        BB_PERIOD + 5
    )
    
    # Track state for smoothing and hysteresis
    prev_signal = 0.0
    prev_direction = 0  # 0=neutral, 1=long, -1=short
    regime_memory = [0] * REGIME_MEMORY  # 1=trending, 0=ranging
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip invalid data
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Check volatility regime (skip extremely low or high volatility)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Determine current regime
        regime = determine_regime(
            adx[i], bb_width[i], ADX_TREND_THRESHOLD, 
            ADX_WEAK_THRESHOLD, BB_SQUEEZE_THRESHOLD
        )
        
        # Update regime memory
        regime_memory.pop(0)
        regime_memory.append(1 if regime == 'trending' else 0)
        recent_trending_ratio = sum(regime_memory) / REGIME_MEMORY
        
        # Calculate EMA stack trend score
        trend_score = calculate_ema_stack_score(
            ema_fast[i], ema_medium[i], ema_slow[i], ema_major[i], close[i]
        )
        
        # Volume confirmation
        volume_spike = volume_ratio[i] >= VOLUME_SPIKE_THRESHOLD
        
        # Initialize raw signal
        raw_signal = 0.0
        regime_weight = 0.0
        
        # =========================================================
        # TRENDING REGIME: Follow the trend with EMA confirmation
        # =========================================================
        if regime == 'trending':
            regime_weight = MIN_SIGNAL_TRENDING
            
            # Base signal from trend score
            raw_signal = trend_score
            
            # Amplify strong trends (ADX > threshold)
            if adx[i] >= ADX_TREND_THRESHOLD:
                raw_signal *= 1.15
            
            # Volume confirmation boost
            if volume_spike:
                raw_signal *= 1.1
            
            # RSI filter: reduce signal if RSI suggests exhaustion
            if trend_score > 0 and rsi[i] > RSI_OVERBOUGHT:
                raw_signal *= 0.7
            elif trend_score < 0 and rsi[i] < RSI_OVERSOLD:
                raw_signal *= 0.7
        
        # =========================================================
        # RANGING REGIME: Mean reversion with RSI
        # =========================================================
        elif regime == 'ranging':
            regime_weight = MIN_SIGNAL_RANGING
            
            # RSI-based mean reversion
            if rsi[i] < RSI_EXTREME_LOW:
                raw_signal = 0.5  # Strong long signal
            elif rsi[i] < RSI_OVERSOLD:
                raw_signal = 0.3  # Moderate long signal
            elif rsi[i] > RSI_EXTREME_HIGH:
                raw_signal = -0.5  # Strong short signal
            elif rsi[i] > RSI_OVERBOUGHT:
                raw_signal = -0.3  # Moderate short signal
            else:
                raw_signal = 0.0  # No signal in neutral RSI
            
            # Reduce signal if fighting the major trend
            if abs(trend_score) > 0.3:
                if np.sign(raw_signal) != np.sign(trend_score):
                    raw_signal *= 0.5  # Reduce counter-trend trades
        
        # =========================================================
        # BREAKOUT REGIME: Squeeze with potential expansion
        # =========================================================
        elif regime == 'breakout':
            regime_weight = MIN_SIGNAL_TRENDING * 1.2  # Higher weight for breakouts
            
            # Need both trend direction AND volume confirmation
            if abs(trend_score) > 0.2 and volume_spike:
                raw_signal = trend_score * 1.2  # Amplify breakout signals
            else:
                # Squeeze without clear direction = wait
                raw_signal = 0.0
                regime_weight = 0.0
        
        # =========================================================
        # TRANSITION REGIME: Conservative, blend signals
        # =========================================================
        else:  # transition
            regime_weight = MIN_SIGNAL_RANGING * 0.7  # Lower weight in uncertainty
            
            # Blend trend and mean reversion based on recent regime history
            if recent_trending_ratio > 0.6:
                # Recently trending, favor trend signals
                raw_signal = trend_score * 0.6
            elif recent_trending_ratio < 0.4:
                # Recently ranging, favor RSI signals
                if rsi[i] < RSI_OVERSOLD:
                    raw_signal = 0.25
                elif rsi[i] > RSI_OVERBOUGHT:
                    raw_signal = -0.25
                else:
                    raw_signal = 0.0
            else:
                # Mixed regime, very conservative
                raw_signal = trend_score * 0.3
        
        # =========================================================
        # VOLATILITY-BASED POSITION SIZING
        # =========================================================
        # Reduce position size in high volatility, increase in low volatility
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)  # Tighter bounds than v2
        
        raw_signal *= vol_factor
        
        # =========================================================
        # SIGNAL SMOOTHING WITH HYSTERESIS
        # =========================================================
        # Exponential smoothing
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Hysteresis: prevent rapid direction flips
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            # Only allow flip if signal change is significant
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