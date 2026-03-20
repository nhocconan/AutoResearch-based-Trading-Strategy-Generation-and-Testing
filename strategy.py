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
    - Funding rate bias integration (mean reversion signal)
    - Enhanced regime transition smoothing (exponential decay memory)
    - Better volatility-adaptive position sizing (ATR-based)
    - Improved momentum confirmation with price action structure
    - RSI divergence with stronger confirmation requirements
    - Volume profile analysis (relative to recent average, not percentile)
    
    Key improvements over adaptive_regime_trend_v2:
    - Funding rate mean reversion: extreme funding → contrarian signal
    - Regime memory with exponential decay (not simple rolling)
    - Volatility-targeted position sizing (more aggressive in low vol)
    - Price action confirmation (higher highs/lower lows)
    - Stronger divergence confirmation (requires multiple bars)
    - Better handling of regime transitions (gradual weight shifts)

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
leverage = 2.8  # Slightly increased due to better risk management

# EMA periods for trend detection
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
ADX_TREND_THRESHOLD = 20
ADX_STRONG_THRESHOLD = 35
ADX_WEAK_THRESHOLD = 15

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
VOLUME_SPIKE_THRESHOLD = 1.5  # Volume > 1.5x average = spike

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012
VOLATILITY_MIN = 0.0015
VOLATILITY_MAX = 0.060

# Signal configuration
MIN_SIGNAL_TRENDING = 0.15
MIN_SIGNAL_RANGING = 0.18
MIN_SIGNAL_BREAKOUT = 0.22
MAX_SIGNAL = 0.85
SMOOTHING_FACTOR = 0.65
HYSTERESIS_THRESHOLD = 0.08

# Regime transition smoothing
REGIME_DECAY = 0.85  # Exponential decay factor for regime memory

# Funding rate configuration (if available)
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
    Calculate volume ratio to rolling average using only past data.
    Returns ratio where >1.0 means above average volume.
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    volume_series = pd.Series(volume)
    avg_volume = volume_series.rolling(window=lookback, min_periods=lookback).mean()
    
    volume_ratio = np.nan_to_num(volume / avg_volume, nan=1.0)
    
    return volume_ratio


def calculate_price_structure(close: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Detect price structure (higher highs/lows or lower highs/lows).
    Returns: positive for bullish structure, negative for bearish, 0 for neutral.
    Uses only past data.
    """
    n = len(close)
    structure = np.zeros(n, dtype=np.float64)
    
    if n < lookback * 2:
        return structure
    
    for i in range(lookback * 2, n):
        recent_high = np.max(close[i-lookback:i])
        prev_high = np.max(close[i-lookback*2:i-lookback])
        recent_low = np.min(close[i-lookback:i])
        prev_low = np.min(close[i-lookback*2:i-lookback])
        
        bullish_signals = 0
        bearish_signals = 0
        
        if recent_high > prev_high * 1.002:
            bullish_signals += 1
        if recent_low > prev_low * 1.002:
            bullish_signals += 1
            
        if recent_high < prev_high * 0.998:
            bearish_signals += 1
        if recent_low < prev_low * 0.998:
            bearish_signals += 1
        
        if bullish_signals > bearish_signals:
            structure[i] = (bullish_signals - bearish_signals) / 2.0
        elif bearish_signals > bullish_signals:
            structure[i] = -(bearish_signals - bullish_signals) / 2.0
    
    return structure


def detect_rsi_divergence(close: np.ndarray, rsi: np.ndarray, lookback: int = 7) -> np.ndarray:
    """
    Detect RSI divergence using only past data.
    Requires stronger confirmation than v2 (multiple bar confirmation).
    Returns: 1 for bullish divergence, -1 for bearish, 0 for none
    """
    n = len(close)
    divergence = np.zeros(n, dtype=np.float64)
    
    if n < lookback * 3:
        return divergence
    
    for i in range(lookback * 3, n):
        # Bullish divergence: price makes lower low, RSI makes higher low
        price_low_1 = np.min(close[i-lookback*2:i-lookback])
        price_low_2 = np.min(close[i-lookback:i])
        rsi_low_1 = np.min(rsi[i-lookback*2:i-lookback])
        rsi_low_2 = np.min(rsi[i-lookback:i])
        
        if price_low_2 < price_low_1 * 0.995 and rsi_low_2 > rsi_low_1 * 1.03:
            divergence[i] = 1.0
            continue
        
        # Bearish divergence: price makes higher high, RSI makes lower high
        price_high_1 = np.max(close[i-lookback*2:i-lookback])
        price_high_2 = np.max(close[i-lookback:i])
        rsi_high_1 = np.max(rsi[i-lookback*2:i-lookback])
        rsi_high_2 = np.max(rsi[i-lookback:i])
        
        if price_high_2 > price_high_1 * 1.005 and rsi_high_2 < rsi_high_1 * 0.97:
            divergence[i] = -1.0
    
    return divergence


def calculate_trend_strength(close: float, ema_fast: float, ema_medium: float, 
                             ema_slow: float, ema_major: float) -> float:
    """
    Calculate trend strength score based on EMA stack alignment.
    Returns value in [-1, 1] where magnitude indicates strength.
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    fast_dev = (ema_fast - ema_medium) / close
    medium_dev = (ema_medium - ema_slow) / close
    slow_dev = (ema_slow - ema_major) / close
    major_dev = (close - ema_major) / close
    
    major_direction = np.sign(major_dev)
    
    if major_direction > 0:
        alignment = 0.0
        if ema_fast > ema_medium:
            alignment += 0.4
        if ema_medium > ema_slow:
            alignment += 0.35
        if ema_slow > ema_major:
            alignment += 0.25
        trend_strength = alignment * major_direction
    else:
        alignment = 0.0
        if ema_fast < ema_medium:
            alignment += 0.4
        if ema_medium < ema_slow:
            alignment += 0.35
        if ema_slow < ema_major:
            alignment += 0.25
        trend_strength = -alignment
    
    avg_dev = abs(fast_dev + medium_dev + slow_dev) / 3
    trend_strength *= np.clip(avg_dev * 100, 0.5, 2.0)
    
    return np.clip(trend_strength, -1.0, 1.0)


def calculate_regime_confidence(adx: float, bb_width: float, adx_trend: float, 
                                 adx_weak: float, bb_squeeze: float) -> tuple:
    """
    Calculate regime confidence scores with smoother transitions.
    Returns: (trending_confidence, ranging_confidence, breakout_potential)
    All values in [0, 1].
    """
    # Trending confidence from ADX with smoother transition
    if adx >= adx_trend:
        trending_conf = np.clip(0.5 + (adx - adx_trend) / 40, 0.5, 1.0)
    elif adx >= adx_weak:
        trending_conf = np.clip((adx - adx_weak) / (adx_trend - adx_weak) * 0.5, 0.2, 0.5)
    else:
        trending_conf = np.clip(adx / adx_weak * 0.2, 0.0, 0.2)
    
    # Ranging confidence (inverse but not perfectly)
    ranging_conf = 1.0 - trending_conf * 0.9
    
    # Breakout potential
    breakout_potential = 0.0
    if bb_width < bb_squeeze:
        breakout_potential = 0.4 + 0.6 * (1 - bb_width / bb_squeeze)
        breakout_potential = np.clip(breakout_potential, 0.0, 1.0)
    
    return trending_conf, ranging_conf, breakout_potential


def calculate_funding_bias(funding_rate: np.ndarray, threshold: float = 0.0005) -> np.ndarray:
    """
    Calculate funding rate bias for mean reversion.
    Extreme positive funding → short bias, extreme negative → long bias.
    Returns values in [-1, 1].
    """
    n = len(funding_rate)
    bias = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        fr = funding_rate[i]
        if fr > threshold:
            # Extreme positive funding → short bias
            bias[i] = -np.clip((fr - threshold) / threshold, 0, 1)
        elif fr < -threshold:
            # Extreme negative funding → long bias
            bias[i] = np.clip((-fr - threshold) / threshold, 0, 1)
    
    return bias


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Regime Trend V3 Strategy.
    
    Signal Logic:
    1. Calculate regime confidence with exponential decay memory
    2. Apply logic weighted by regime confidence
    3. Bollinger Band squeeze detection for breakout preparation
    4. MACD + RSI momentum confirmation with price structure
    5. Volume confirmation (ratio to average, not percentile)
    6. Funding rate bias for mean reversion (if available)
    7. Volatility-adaptive position sizing
    8. Signal smoothing with hysteresis and regime memory
    
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
    
    # Check for funding rate data (optional)
    funding_rate = None
    funding_bias = np.zeros(n, dtype=np.float64)
    if "funding_rate" in prices.columns:
        try:
            funding_rate = prices["funding_rate"].values.astype(np.float64)
            funding_rate = np.nan_to_num(funding_rate, nan=0.0)
            funding_bias = calculate_funding_bias(funding_rate, FUNDING_EXTREME_THRESHOLD)
        except (KeyError, TypeError, ValueError):
            funding_rate = None
    
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
    price_structure = calculate_price_structure(close, lookback=5)
    rsi_divergence = detect_rsi_divergence(close, rsi, lookback=7)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        ADX_PERIOD * 2 + 1,
        VOLUME_LOOKBACK,
        BB_PERIOD,
        MACD_SLOW + MACD_SIGNAL,
        21  # For price structure
    )
    
    # Track state for smoothing and hysteresis
    prev_signal = 0.0
    prev_direction = 0
    regime_memory = 0.5  # Exponential decay memory (0=ranging, 1=trending)
    
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
        
        # Update regime memory with exponential decay
        current_regime = trending_conf
        regime_memory = REGIME_DECAY * regime_memory + (1 - REGIME_DECAY) * current_regime
        
        # Determine dominant regime
        is_squeeze = bb_width[i] < BB_SQUEEZE_THRESHOLD
        is_trending = regime_memory >= 0.5
        is_ranging = regime_memory < 0.5
        
        # Calculate trend strength
        trend_strength = calculate_trend_strength(
            close[i], ema_fast[i], ema_medium[i],
            ema_slow[i], ema_major[i]
        )
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_SPIKE_THRESHOLD
        
        # Initialize raw signal and regime weight
        raw_signal = 0.0
        regime_weight = 0.0
        
        # BREAKOUT REGIME (squeeze + potential expansion)
        if is_squeeze and breakout_potential > 0.3:
            if trend_strength > 0.15 and (volume_confirmed or price_structure[i] > 0):
                raw_signal = trend_strength * (0.5 + breakout_potential * 0.5)
                regime_weight = MIN_SIGNAL_BREAKOUT
            elif trend_strength < -0.15 and (volume_confirmed or price_structure[i] < 0):
                raw_signal = trend_strength * (0.5 + breakout_potential * 0.5)
                regime_weight = MIN_SIGNAL_BREAKOUT
            else:
                raw_signal = 0.0
                regime_weight = 0.0
        
        # TRENDING REGIME
        elif is_trending:
            regime_weight = MIN_SIGNAL_TRENDING
            raw_signal = trend_strength
            
            # Amplify in strong trends
            if adx[i] >= ADX_STRONG_THRESHOLD:
                raw_signal *= 1.15
            
            # Price structure confirmation
            if trend_strength > 0 and price_structure[i] > 0:
                raw_signal *= 1.1
            elif trend_strength < 0 and price_structure[i] < 0:
                raw_signal *= 1.1
            
            # Volume boost
            if volume_confirmed:
                raw_signal *= 1.08
            
            # MACD confirmation
            macd_conf = np.clip(np.sign(macd_hist[i]) * trend_strength, 0, 1)
            raw_signal *= (0.75 + 0.25 * macd_conf)
            
            # RSI divergence reduces trend signal
            if rsi_divergence[i] != 0:
                raw_signal *= 0.5
        
        # RANGING REGIME
        elif is_ranging:
            regime_weight = MIN_SIGNAL_RANGING
            
            # RSI-based mean reversion
            if rsi[i] < RSI_OVERSOLD:
                raw_signal = 0.35
            elif rsi[i] > RSI_OVERBOUGHT:
                raw_signal = -0.35
            else:
                raw_signal = 0.0
            
            # Amplify at extremes
            if rsi[i] < RSI_EXTREME_LOW:
                raw_signal = 0.55
            elif rsi[i] > RSI_EXTREME_HIGH:
                raw_signal = -0.55
            
            # RSI divergence confirmation
            if rsi_divergence[i] == 1.0:
                raw_signal = max(raw_signal, 0.35)
            elif rsi_divergence[i] == -1.0:
                raw_signal = min(raw_signal, -0.35)
            
            # Reduce if trend is strong against mean reversion
            if abs(trend_strength) > 0.25:
                raw_signal *= 0.6
        
        # TRANSITION REGIME
        else:
            regime_weight = MIN_SIGNAL_RANGING * 0.85
            
            trend_signal = trend_strength * 0.5
            rsi_signal = 0.0
            if rsi[i] < RSI_OVERSOLD:
                rsi_signal = 0.25
            elif rsi[i] > RSI_OVERBOUGHT:
                rsi_signal = -0.25
            
            raw_signal = regime_memory * trend_signal + (1 - regime_memory) * rsi_signal
        
        # Apply funding rate bias (mean reversion signal)
        if funding_rate is not None and abs(funding_bias[i]) > 0.1:
            # Funding bias acts as contrarian signal
            if raw_signal * funding_bias[i] < 0:
                # Funding agrees with signal direction
                raw_signal += funding_bias[i] * FUNDING_BIAS_WEIGHT
            else:
                # Funding disagrees, reduce signal
                raw_signal *= (1 - abs(funding_bias[i]) * FUNDING_BIAS_WEIGHT * 0.5)
        
        # Volatility-based position sizing
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.2)
        
        raw_signal *= vol_factor
        
        # Apply exponential smoothing
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Apply hysteresis
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