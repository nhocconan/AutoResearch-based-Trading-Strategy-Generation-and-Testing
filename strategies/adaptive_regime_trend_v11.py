#!/usr/bin/env python3
"""
strategy.py - Adaptive Regime Trend V11
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on adaptive_regime_trend_v10 with critical bug fixes:
    - Fixed is_ranging undefined variable bug
    - Simplified regime detection for better stability
    - Enhanced funding rate mean reversion with adaptive thresholds
    - Multi-layer EMA trend confirmation (3-tier stack)
    - Volume confirmation scaled by volatility regime
    - Adaptive signal smoothing based on regime stability
    - Improved hysteresis to reduce whipsaws during transitions
    
    Key improvements over adaptive_regime_trend_v10:
    - Fixed critical bug: is_ranging now properly defined as not is_trending
    - Cleaner regime logic with explicit binary classification
    - Funding threshold adapts to both volatility and recent extremes
    - EMA stack uses 3-tier confirmation (fast/medium/slow alignment)
    - Signal smoothing factor adapts to regime stability
    - Hysteresis threshold scales with signal magnitude
    - Reduced parameter count for better generalization

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

name = "adaptive_regime_trend_v11"
timeframe = "1h"
leverage = 2.5  # Moderate leverage for risk-adjusted returns

# EMA periods for trend detection (3-tier stack)
EMA_FAST = 12
EMA_MEDIUM = 26
EMA_SLOW = 50
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

# Bollinger Band configuration
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_THRESHOLD = 0.02

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_THRESHOLD = 1.5
VOLUME_SPIKE_VOL_SCALE = 0.3

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.010
VOLATILITY_MIN = 0.002
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL_TRENDING = 0.20
MIN_SIGNAL_RANGING = 0.25
MAX_SIGNAL = 0.80
BASE_SMOOTHING = 0.60
SMOOTHING_ADAPTIVE_RANGE = 0.15
HYSTERESIS_BASE = 0.08
HYSTERESIS_SIGNAL_SCALE = 0.5

# Funding rate configuration
FUNDING_BASE_THRESHOLD = 0.0005
FUNDING_BIAS_WEIGHT = 0.25
FUNDING_VOL_LOOKBACK = 50
FUNDING_EXTREME_LOOKBACK = 100
FUNDING_EXTREME_SCALE = 2.0

# Taker ratio configuration
TAKER_RATIO_THRESHOLD = 0.55
TAKER_BIAS_WEIGHT = 0.20

# Regime stability tracking
REGIME_STABILITY_LOOKBACK = 10
REGIME_STABILITY_THRESHOLD = 0.6


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


def calculate_volume_spike(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Detect volume spikes using rolling average.
    Returns ratio of current volume to rolling average.
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


def calculate_funding_extremes(funding_rate: np.ndarray, lookback: int = 100) -> tuple:
    """
    Calculate rolling extremes of funding rate for adaptive threshold.
    Returns: (rolling_max, rolling_min)
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    rolling_max = np.zeros(n, dtype=np.float64)
    rolling_min = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return rolling_max, rolling_min
    
    funding_series = pd.Series(funding_rate)
    rolling_max_series = funding_series.rolling(window=lookback, min_periods=lookback).max()
    rolling_min_series = funding_series.rolling(window=lookback, min_periods=lookback).min()
    
    rolling_max = np.nan_to_num(rolling_max_series.values, nan=0.0)
    rolling_min = np.nan_to_num(rolling_min_series.values, nan=0.0)
    
    return rolling_max, rolling_min


def calculate_funding_volatility(funding_rate: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calculate rolling volatility of funding rate for dynamic threshold.
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    funding_vol = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return funding_vol
    
    funding_series = pd.Series(funding_rate)
    funding_vol_series = funding_series.rolling(window=lookback, min_periods=lookback).std()
    
    funding_vol = np.nan_to_num(funding_vol_series.values, nan=0.0)
    
    return funding_vol


def calculate_funding_bias(funding_rate: np.ndarray, funding_vol: np.ndarray,
                           funding_max: np.ndarray, funding_min: np.ndarray,
                           base_threshold: float = 0.0005, extreme_scale: float = 2.0) -> np.ndarray:
    """
    Calculate funding rate bias for mean reversion signal with adaptive threshold.
    Extreme positive funding → short bias
    Extreme negative funding → long bias
    Returns value in [-1, 1].
    Only uses current/past funding rate (no look-ahead).
    """
    n = len(funding_rate)
    bias = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        # Adaptive threshold based on funding volatility and recent extremes
        vol_adjustment = 1.0 + funding_vol[i] * 1000
        extreme_range = max(abs(funding_max[i]), abs(funding_min[i]), base_threshold)
        dynamic_threshold = base_threshold * vol_adjustment * extreme_scale
        dynamic_threshold = max(dynamic_threshold, extreme_range * 0.5)
        
        if funding_rate[i] > dynamic_threshold:
            bias[i] = -np.clip(funding_rate[i] / dynamic_threshold, 0, 1)
        elif funding_rate[i] < -dynamic_threshold:
            bias[i] = np.clip(-funding_rate[i] / dynamic_threshold, 0, 1)
        else:
            bias[i] = 0.0
    
    return bias


def calculate_taker_bias(taker_ratio: np.ndarray, threshold: float = 0.55) -> np.ndarray:
    """
    Calculate taker buy/sell ratio bias.
    High taker buy ratio → long bias
    Low taker buy ratio → short bias
    Returns value in [-1, 1].
    Only uses current/past taker ratio (no look-ahead).
    """
    n = len(taker_ratio)
    bias = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if taker_ratio[i] > threshold:
            bias[i] = np.clip((taker_ratio[i] - threshold) / (1.0 - threshold), 0, 1)
        elif taker_ratio[i] < (1.0 - threshold):
            bias[i] = -np.clip((threshold - taker_ratio[i]) / threshold, 0, 1)
        else:
            bias[i] = 0.0
    
    return bias


def calculate_trend_strength(close: float, ema_fast: float, ema_medium: float, 
                             ema_slow: float, ema_major: float) -> float:
    """
    Calculate trend strength score based on EMA stack alignment.
    Returns value in [-1, 1] where magnitude indicates strength.
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    # Calculate deviations
    fast_dev = (ema_fast - ema_medium) / close
    medium_dev = (ema_medium - ema_slow) / close
    slow_dev = (ema_slow - ema_major) / close
    major_dev = (close - ema_major) / close
    
    # Determine primary direction from major EMA
    major_direction = np.sign(major_dev)
    
    if major_direction > 0:
        # Bullish alignment check
        alignment = 0.0
        if ema_fast > ema_medium:
            alignment += 0.35
        if ema_medium > ema_slow:
            alignment += 0.35
        if ema_slow > ema_major:
            alignment += 0.30
        trend_strength = alignment
    elif major_direction < 0:
        # Bearish alignment check
        alignment = 0.0
        if ema_fast < ema_medium:
            alignment += 0.35
        if ema_medium < ema_slow:
            alignment += 0.35
        if ema_slow < ema_major:
            alignment += 0.30
        trend_strength = -alignment
    else:
        trend_strength = 0.0
    
    # Scale by average deviation magnitude
    avg_dev = abs(fast_dev + medium_dev + slow_dev) / 3
    trend_strength *= np.clip(avg_dev * 100, 0.5, 2.0)
    
    return np.clip(trend_strength, -1.0, 1.0)


def calculate_regime_stability(regime_history: list, lookback: int) -> float:
    """
    Calculate regime stability score.
    Higher value = more stable regime (less switching).
    Returns value in [0, 1].
    """
    if len(regime_history) < lookback:
        return 0.5
    
    recent = regime_history[-lookback:]
    same_count = sum(1 for i in range(1, len(recent)) if recent[i] == recent[i-1])
    stability = same_count / (len(recent) - 1) if len(recent) > 1 else 0.5
    
    return stability


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Regime Trend V11 Strategy.
    
    Signal Logic:
    1. Calculate regime (trending vs ranging) using ADX + BB width
    2. Trending: Follow EMA stack direction with momentum confirmation
    3. Ranging: Mean reversion using RSI extremes
    4. Funding rate bias as contrarian indicator (adaptive threshold)
    5. Taker ratio for market pressure confirmation
    6. Volume confirmation scaled by volatility
    7. Adaptive signal smoothing based on regime stability
    8. Hysteresis to reduce whipsaws during transitions
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, funding_rate, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
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
        
        try:
            taker_ratio = prices["taker_buy_volume"].values.astype(np.float64) / \
                         (prices["volume"].values.astype(np.float64) + 1e-10)
            taker_ratio = np.nan_to_num(taker_ratio, nan=0.5)
        except (KeyError, TypeError, ValueError):
            taker_ratio = np.full(n, 0.5, dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return signals
    
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_medium = calculate_ema(close, EMA_MEDIUM)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    
    volume_ratio = calculate_volume_spike(volume, VOLUME_LOOKBACK)
    funding_vol = calculate_funding_volatility(funding_rate, FUNDING_VOL_LOOKBACK)
    funding_max, funding_min = calculate_funding_extremes(funding_rate, FUNDING_EXTREME_LOOKBACK)
    funding_bias = calculate_funding_bias(funding_rate, funding_vol, funding_max, funding_min,
                                          FUNDING_BASE_THRESHOLD, FUNDING_EXTREME_SCALE)
    taker_bias = calculate_taker_bias(taker_ratio, TAKER_RATIO_THRESHOLD)
    
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        ADX_PERIOD * 2 + 1,
        VOLUME_LOOKBACK,
        BB_PERIOD,
        FUNDING_VOL_LOOKBACK,
        FUNDING_EXTREME_LOOKBACK
    )
    
    prev_signal = 0.0
    prev_direction = 0
    regime_history = [0] * REGIME_STABILITY_LOOKBACK
    
    for i in range(min_valid_index, n):
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        is_trending = adx[i] >= ADX_TREND_THRESHOLD
        is_squeeze = bb_width[i] < BB_SQUEEZE_THRESHOLD
        is_ranging = not is_trending  # FIXED: was undefined in v10
        
        regime_history.pop(0)
        regime_history.append(1 if is_trending else 0)
        regime_stability = calculate_regime_stability(regime_history, REGIME_STABILITY_LOOKBACK)
        
        trend_strength = calculate_trend_strength(
            close[i], ema_fast[i], ema_medium[i],
            ema_slow[i], ema_major[i]
        )
        
        vol_adjusted_threshold = VOLUME_SPIKE_THRESHOLD * (1.0 + VOLUME_SPIKE_VOL_SCALE * atr_pct * 100)
        volume_confirmed = volume_ratio[i] >= vol_adjusted_threshold
        
        raw_signal = 0.0
        regime_weight = 0.0
        
        if is_trending and not is_squeeze:
            regime_weight = MIN_SIGNAL_TRENDING
            raw_signal = trend_strength
            
            if adx[i] >= ADX_STRONG_THRESHOLD:
                raw_signal *= 1.15
            
            if volume_confirmed:
                raw_signal *= 1.1
            
            if abs(funding_bias[i]) > 0.3:
                if (trend_strength > 0 and funding_bias[i] < 0) or \
                   (trend_strength < 0 and funding_bias[i] > 0):
                    raw_signal *= (1.0 - FUNDING_BIAS_WEIGHT * 0.5)
            
            if abs(taker_bias[i]) > 0.3:
                if (trend_strength > 0 and taker_bias[i] > 0) or \
                   (trend_strength < 0 and taker_bias[i] < 0):
                    raw_signal *= (1.0 + TAKER_BIAS_WEIGHT * 0.5)
        
        elif is_ranging or is_squeeze:
            regime_weight = MIN_SIGNAL_RANGING
            
            if rsi[i] < RSI_OVERSOLD:
                raw_signal = 0.35 + 0.25 * ((RSI_OVERSOLD - rsi[i]) / RSI_OVERSOLD)
            elif rsi[i] > RSI_OVERBOUGHT:
                raw_signal = -0.35 - 0.25 * ((rsi[i] - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT))
            else:
                raw_signal = 0.0
            
            if rsi[i] < RSI_EXTREME_LOW:
                raw_signal = min(raw_signal, 0.6)
            elif rsi[i] > RSI_EXTREME_HIGH:
                raw_signal = max(raw_signal, -0.6)
            
            if abs(funding_bias[i]) > 0.3:
                if (raw_signal > 0 and funding_bias[i] > 0) or \
                   (raw_signal < 0 and funding_bias[i] < 0):
                    raw_signal *= (1.0 + FUNDING_BIAS_WEIGHT)
            
            if abs(trend_strength) > 0.3:
                raw_signal *= 0.6
            
            if abs(taker_bias[i]) > 0.3:
                if (raw_signal > 0 and taker_bias[i] > 0) or \
                   (raw_signal < 0 and taker_bias[i] < 0):
                    raw_signal *= (1.0 + TAKER_BIAS_WEIGHT * 0.5)
        
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 1.8)
        
        raw_signal *= vol_factor
        
        smoothing_factor = BASE_SMOOTHING + SMOOTHING_ADAPTIVE_RANGE * regime_stability
        smoothed_signal = smoothing_factor * prev_signal + (1.0 - smoothing_factor) * raw_signal
        
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            hysteresis_threshold = HYSTERESIS_BASE + HYSTERESIS_SIGNAL_SCALE * abs(raw_signal)
            if abs(smoothed_signal - prev_signal) < hysteresis_threshold:
                smoothed_signal = prev_signal
        
        if abs(smoothed_signal) < regime_weight:
            smoothed_signal = 0.0
        
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals