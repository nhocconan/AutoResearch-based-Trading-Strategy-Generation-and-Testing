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
    - Longer regime memory for smoother transitions (reduce whipsaws)
    - Volume ratio confirmation instead of percentile (more robust)
    - Funding rate bias when available (crowd positioning signal)
    - Adaptive smoothing based on regime stability
    - Cleaner EMA ribbon trend strength calculation
    - Better volatility-based position sizing
    
    Key improvements over adaptive_regime_trend_v2:
    - Regime memory extended to 8 bars for smoother transitions
    - Volume ratio: current vs average (more intuitive than percentile)
    - Funding rate mean reversion signal (if data available)
    - Adaptive smoothing: more smoothing in unstable regimes
    - EMA ribbon alignment score (all 4 EMAs)
    - Dynamic volatility target based on recent ATR

Look-Ahead Safety:
    - All rolling calculations use only past data (min_periods respected)
    - No .shift(-n) or future index access
    - Signal at bar t uses only prices.iloc[:t+1]
    - Funding rate accessed only from current/past rows
"""

import numpy as np
import pandas as pd

# =============================================================================
# Strategy Configuration
# =============================================================================

name = "adaptive_regime_trend_v3"
timeframe = "1h"
leverage = 2.5  # Conservative leverage for crypto futures

# EMA periods for trend detection (ribbon)
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
ADX_STRONG_THRESHOLD = 40
ADX_WEAK_THRESHOLD = 20

# Bollinger Band configuration
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_THRESHOLD = 0.012

# MACD configuration
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_RATIO_THRESHOLD = 1.3  # Current volume / avg volume

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.008
VOLATILITY_MIN = 0.0015
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL_TRENDING = 0.15
MIN_SIGNAL_RANGING = 0.25
MIN_SIGNAL_BREAKOUT = 0.30
MAX_SIGNAL = 0.75
BASE_SMOOTHING = 0.65
ADAPTIVE_SMOOTHING_RANGE = 0.15
HYSTERESIS_THRESHOLD = 0.08

# Regime transition smoothing
REGIME_MEMORY = 8  # Extended memory for smoother transitions

# Funding rate configuration
FUNDING_EXTREME_THRESHOLD = 0.0005  # 0.05% per 8h
FUNDING_BIAS_WEIGHT = 0.15


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
        
        if plus_dm[i] > minus_dm[i]:
            minus_dm[i] = 0.0
        elif minus_dm[i] > plus_dm[i]:
            plus_dm[i] = 0.0
    
    tr_series = pd.Series(tr)
    plus_dm_series = pd.Series(plus_dm)
    minus_dm_series = pd.Series(minus_dm)
    
    atr_series = tr_series.ewm(span=period, adjust=False, min_periods=period).mean()
    plus_di_series = (plus_dm_series.ewm(span=period, adjust=False, min_periods=period).mean() / 
                      atr_series * 100)
    minus_di_series = (minus_dm_series.ewm(span=period, adjust=False, min_periods=period).mean() / 
                       atr_series * 100)
    
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
    Calculate volume ratio: current volume / rolling average volume.
    Only uses past volume data (no look-ahead).
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=lookback, min_periods=lookback).mean()
    
    volume_ratio = np.nan_to_num((volume_series / avg_volume).values, nan=1.0)
    
    return volume_ratio


def calculate_ema_ribbon_score(ema_fast: float, ema_medium: float, 
                                ema_slow: float, ema_major: float,
                                close: float) -> float:
    """
    Calculate EMA ribbon alignment score.
    Returns value in [-1, 1] where magnitude indicates alignment strength.
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    # Check bullish alignment: close > fast > medium > slow > major
    bullish = (close > ema_fast > ema_medium > ema_slow > ema_major)
    bearish = (close < ema_fast < ema_medium < ema_slow < ema_major)
    
    if bullish:
        # Calculate alignment strength from spacing
        spacing = (
            (close - ema_fast) / close +
            (ema_fast - ema_medium) / close +
            (ema_medium - ema_slow) / close +
            (ema_slow - ema_major) / close
        )
        score = np.clip(spacing * 25, 0.3, 1.0)
    elif bearish:
        spacing = (
            (ema_fast - close) / close +
            (ema_medium - ema_fast) / close +
            (ema_slow - ema_medium) / close +
            (ema_major - ema_slow) / close
        )
        score = -np.clip(spacing * 25, 0.3, 1.0)
    else:
        # Mixed alignment - calculate net bias
        above_count = sum([close > ema_fast, ema_fast > ema_medium, 
                          ema_medium > ema_slow, ema_slow > ema_major])
        below_count = 4 - above_count
        
        if above_count > below_count:
            score = (above_count - below_count) / 4 * 0.5
        else:
            score = -(below_count - above_count) / 4 * 0.5
    
    return np.clip(score, -1.0, 1.0)


def calculate_regime_confidence(adx: float, bb_width: float, 
                                 adx_trend: float, adx_weak: float, 
                                 bb_squeeze: float) -> tuple:
    """
    Calculate regime confidence scores.
    Returns: (trending_confidence, ranging_confidence, breakout_potential)
    All values in [0, 1].
    """
    # Trending confidence from ADX
    if adx >= adx_trend:
        trending_conf = np.clip((adx - adx_trend) / 35 + 0.5, 0.5, 1.0)
    elif adx >= adx_weak:
        trending_conf = np.clip((adx - adx_weak) / (adx_trend - adx_weak) * 0.5, 0.2, 0.5)
    else:
        trending_conf = np.clip(adx / adx_weak * 0.2, 0.0, 0.2)
    
    # Ranging confidence
    ranging_conf = 1.0 - trending_conf
    if bb_width < bb_squeeze:
        ranging_conf *= 0.6  # Squeeze reduces ranging confidence
    
    # Breakout potential
    breakout_potential = 0.0
    if bb_width < bb_squeeze:
        breakout_potential = 0.5 + 0.5 * (1 - bb_width / bb_squeeze)
    breakout_potential = np.clip(breakout_potential * (1 + adx / 50), 0.0, 1.0)
    
    return trending_conf, ranging_conf, breakout_potential


def calculate_funding_bias(funding_rates: np.ndarray, threshold: float) -> np.ndarray:
    """
    Calculate funding rate mean reversion bias.
    Extreme positive funding → short bias (crowd too long)
    Extreme negative funding → long bias (crowd too short)
    Returns value in [-1, 1].
    """
    n = len(funding_rates)
    bias = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if funding_rates[i] > threshold:
            # Extreme positive funding → short bias
            bias[i] = -np.clip((funding_rates[i] - threshold) / threshold, 0, 1)
        elif funding_rates[i] < -threshold:
            # Extreme negative funding → long bias
            bias[i] = np.clip((-funding_rates[i] - threshold) / threshold, 0, 1)
    
    return bias


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Regime Trend V3 Strategy.
    
    Signal Logic:
    1. Calculate regime confidence (trending/ranging/breakout)
    2. Apply logic weighted by regime confidence
    3. Bollinger Band squeeze detection for breakout preparation
    4. MACD + RSI momentum confirmation
    5. Volume ratio confirmation for breakouts
    6. Funding rate bias (if available)
    7. Adaptive signal smoothing based on regime stability
    8. EMA ribbon trend strength
    
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
    
    # Try to get funding rate if available
    funding_rates = None
    funding_bias = None
    try:
        if "funding_rate" in prices.columns:
            funding_rates = prices["funding_rate"].values.astype(np.float64)
            funding_rates = np.nan_to_num(funding_rates, nan=0.0)
            funding_bias = calculate_funding_bias(funding_rates, FUNDING_EXTREME_THRESHOLD)
    except (KeyError, TypeError, ValueError):
        pass
    
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
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        ADX_PERIOD * 2 + 1,
        VOLUME_LOOKBACK,
        BB_PERIOD,
        MACD_SLOW + MACD_SIGNAL
    )
    
    # Track state for smoothing and hysteresis
    prev_signal = 0.0
    prev_direction = 0
    regime_memory = [0] * REGIME_MEMORY
    regime_stability = 0.5
    
    # Calculate recent ATR for dynamic volatility target
    recent_atr_pct = np.zeros(n, dtype=np.float64)
    for i in range(ATR_PERIOD, n):
        recent_atr_pct[i] = np.mean(atr[i-ATR_PERIOD:i] / close[i-ATR_PERIOD:i])
    
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
        
        # Calculate regime confidence
        trending_conf, ranging_conf, breakout_potential = calculate_regime_confidence(
            adx[i], bb_width[i], ADX_TREND_THRESHOLD, ADX_WEAK_THRESHOLD, BB_SQUEEZE_THRESHOLD
        )
        
        # Determine dominant regime
        is_squeeze = bb_width[i] < BB_SQUEEZE_THRESHOLD
        is_trending = trending_conf >= 0.5
        is_ranging = ranging_conf >= 0.5
        
        # Update regime memory and calculate stability
        current_regime = 1 if is_trending else 0
        regime_memory.pop(0)
        regime_memory.append(current_regime)
        regime_stability = 1.0 - np.std(regime_memory)  # Higher = more stable
        
        # Calculate EMA ribbon trend strength
        ribbon_score = calculate_ema_ribbon_score(
            ema_fast[i], ema_medium[i], ema_slow[i], ema_major[i], close[i]
        )
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_RATIO_THRESHOLD
        
        # Initialize raw signal and regime weight
        raw_signal = 0.0
        regime_weight = 0.0
        
        # BREAKOUT REGIME (squeeze + potential expansion)
        if is_squeeze and breakout_potential > 0.3:
            if ribbon_score > 0.25 and volume_confirmed:
                raw_signal = ribbon_score * (0.5 + breakout_potential * 0.5)
                regime_weight = MIN_SIGNAL_BREAKOUT
            elif ribbon_score < -0.25 and volume_confirmed:
                raw_signal = ribbon_score * (0.5 + breakout_potential * 0.5)
                regime_weight = MIN_SIGNAL_BREAKOUT
            else:
                raw_signal = 0.0
                regime_weight = 0.0
        
        # TRENDING REGIME
        elif is_trending:
            regime_weight = MIN_SIGNAL_TRENDING
            raw_signal = ribbon_score
            
            # Amplify in strong trends
            if adx[i] >= ADX_STRONG_THRESHOLD:
                raw_signal *= 1.15
            
            # MACD momentum confirmation
            macd_confirm = np.sign(macd_hist[i])
            if np.sign(raw_signal) == macd_confirm:
                raw_signal *= 1.1
            
            # Volume boost
            if volume_confirmed:
                raw_signal *= 1.05
            
            # RSI divergence check (reduce if conflicting)
            rsi_neutral = 35 < rsi[i] < 65
            if not rsi_neutral and np.sign(raw_signal) != np.sign(50 - rsi[i]):
                raw_signal *= 0.6
        
        # RANGING REGIME
        elif is_ranging:
            regime_weight = MIN_SIGNAL_RANGING
            
            # RSI-based mean reversion
            if rsi[i] < RSI_OVERSOLD:
                raw_signal = 0.35 + (RSI_OVERSOLD - rsi[i]) / 100
            elif rsi[i] > RSI_OVERBOUGHT:
                raw_signal = -0.35 - (rsi[i] - RSI_OVERBOUGHT) / 100
            else:
                raw_signal = 0.0
            
            # Amplify at extremes
            if rsi[i] < RSI_EXTREME_LOW:
                raw_signal = min(raw_signal, -0.5) * -1
            elif rsi[i] > RSI_EXTREME_HIGH:
                raw_signal = max(raw_signal, 0.5) * -1
            
            # Reduce if fighting strong ribbon trend
            if abs(ribbon_score) > 0.4:
                raw_signal *= 0.5
        
        # TRANSITION REGIME
        else:
            regime_weight = MIN_SIGNAL_RANGING * 0.7
            trend_component = ribbon_score * trending_conf * 0.6
            rsi_component = 0.0
            if rsi[i] < RSI_OVERSOLD:
                rsi_component = 0.25 * ranging_conf
            elif rsi[i] > RSI_OVERBOUGHT:
                rsi_component = -0.25 * ranging_conf
            
            raw_signal = trend_component + rsi_component
        
        # Apply funding rate bias if available
        if funding_bias is not None and funding_bias[i] != 0:
            # Funding bias acts as a contrarian signal
            if np.sign(raw_signal) == np.sign(funding_bias[i]):
                raw_signal *= (1 + FUNDING_BIAS_WEIGHT)
            else:
                raw_signal *= (1 - FUNDING_BIAS_WEIGHT)
        
        # Dynamic volatility-based position sizing
        dynamic_vol_target = VOLATILITY_TARGET * (1 + 0.3 * (1 - regime_stability))
        vol_factor = dynamic_vol_target / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        
        raw_signal *= vol_factor
        
        # Adaptive smoothing based on regime stability
        adaptive_smoothing = BASE_SMOOTHING + ADAPTIVE_SMOOTHING_RANGE * (1 - regime_stability)
        adaptive_smoothing = np.clip(adaptive_smoothing, 0.5, 0.85)
        
        smoothed_signal = adaptive_smoothing * prev_signal + (1.0 - adaptive_smoothing) * raw_signal
        
        # Apply hysteresis to reduce flipping
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < HYSTERESIS_THRESHOLD:
                smoothed_signal = prev_signal
        
        # Apply minimum signal threshold
        if abs(smoothed_signal) < regime_weight:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals