#!/usr/bin/env python3
"""
strategy.py - Adaptive Regime Trend V2.1
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on adaptive_regime_trend_v2 success (Sharpe=0.500), improving:
    - Smoother regime transitions with weighted blending
    - Enhanced MACD momentum with histogram slope detection
    - Volume-weighted signal confidence
    - Adaptive hysteresis based on volatility regime
    - Better RSI extreme handling with trend filter
    
    Key improvements over adaptive_regime_trend_v2:
    - Regime blending: smooth transitions instead of hard switches
    - MACD histogram slope for momentum acceleration
    - Volume confidence multiplier on signal strength
    - Volatility-adaptive hysteresis threshold
    - Trend-aligned RSI mean reversion (only trade RSI extremes with trend)

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

name = "adaptive_regime_trend_v2_1"
timeframe = "1h"
leverage = 2.5  # Conservative leverage for crypto futures

# EMA periods for trend detection
EMA_FAST = 9
EMA_MEDIUM = 21
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
VOLUME_PERCENTILE_THRESHOLD = 0.55

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.008
VOLATILITY_MIN = 0.0015
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL_TRENDING = 0.10
MIN_SIGNAL_RANGING = 0.15
MIN_SIGNAL_BREAKOUT = 0.20
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.75
HYSTERESIS_BASE = 0.05

# Regime transition smoothing
REGIME_MEMORY = 7


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


def calculate_macd_slope(macd_hist: np.ndarray, lookback: int = 3) -> np.ndarray:
    """
    Calculate MACD histogram slope (momentum acceleration).
    Positive slope = increasing bullish momentum.
    """
    n = len(macd_hist)
    slope = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return slope
    
    for i in range(lookback, n):
        slope[i] = (macd_hist[i] - macd_hist[i-lookback]) / lookback
    
    return slope


def calculate_volume_percentile(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume percentile rank using rolling window.
    """
    n = len(volume)
    volume_pct = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return volume_pct
    
    volume_series = pd.Series(volume)
    
    for i in range(lookback, n):
        window = volume_series.iloc[i-lookback:i]
        current_vol = volume[i]
        rank = (window < current_vol).sum() / lookback
        volume_pct[i] = rank
    
    return volume_pct


def calculate_trend_strength(close: float, ema_fast: float, ema_medium: float, 
                             ema_slow: float, ema_major: float) -> float:
    """
    Calculate trend strength score based on EMA stack alignment.
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


def calculate_regime_weights(adx: float, bb_width: float, adx_trend: float, 
                             adx_weak: float, bb_squeeze: float) -> tuple:
    """
    Calculate smooth regime weights (trending, ranging, breakout).
    All values sum to 1.0, allowing smooth blending.
    """
    # Trending weight from ADX
    if adx >= adx_trend:
        trending_weight = np.clip(0.5 + (adx - adx_trend) / 40, 0.5, 1.0)
    elif adx >= adx_weak:
        trending_weight = np.clip((adx - adx_weak) / (adx_trend - adx_weak) * 0.5, 0.2, 0.5)
    else:
        trending_weight = np.clip(adx / adx_weak * 0.2, 0.0, 0.2)
    
    # Breakout weight from BB squeeze
    breakout_weight = 0.0
    if bb_width < bb_squeeze:
        breakout_weight = 0.3 + 0.7 * (1 - bb_width / bb_squeeze)
        breakout_weight = np.clip(breakout_weight, 0.3, 0.8)
    
    # Ranging weight is remainder
    ranging_weight = 1.0 - trending_weight - breakout_weight * 0.5
    ranging_weight = np.clip(ranging_weight, 0.0, 1.0)
    
    # Normalize
    total = trending_weight + ranging_weight + breakout_weight
    if total > 0:
        trending_weight /= total
        ranging_weight /= total
        breakout_weight /= total
    
    return trending_weight, ranging_weight, breakout_weight


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Regime Trend V2.1 Strategy.
    
    Signal Logic:
    1. Calculate smooth regime weights (trending/ranging/breakout)
    2. Blend signals based on regime weights
    3. MACD histogram slope for momentum acceleration
    4. Volume confidence multiplier
    5. Volatility-adaptive hysteresis
    6. Trend-aligned RSI mean reversion
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, ...]
    
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
    except (KeyError, TypeError, ValueError):
        return signals
    
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
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
    macd_slope = calculate_macd_slope(macd_hist, lookback=3)
    
    volume_pct = calculate_volume_percentile(volume, VOLUME_LOOKBACK)
    
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        ADX_PERIOD * 2 + 1,
        VOLUME_LOOKBACK,
        BB_PERIOD,
        MACD_SLOW + MACD_SIGNAL + 3
    )
    
    prev_signal = 0.0
    prev_direction = 0
    regime_memory = [0] * REGIME_MEMORY
    
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
        
        # Calculate smooth regime weights
        trending_w, ranging_w, breakout_w = calculate_regime_weights(
            adx[i], bb_width[i], ADX_TREND_THRESHOLD, ADX_WEAK_THRESHOLD, BB_SQUEEZE_THRESHOLD
        )
        
        # Update regime memory
        regime_memory.pop(0)
        regime_memory.append(1 if trending_w > 0.5 else 0)
        recent_trending = sum(regime_memory) / REGIME_MEMORY
        
        # Calculate trend strength
        trend_strength = calculate_trend_strength(
            close[i], ema_fast[i], ema_medium[i],
            ema_slow[i], ema_major[i]
        )
        
        # Volume confidence
        volume_conf = np.clip(volume_pct[i] * 1.5, 0.5, 1.3)
        
        # Momentum score with MACD slope
        momentum_score = 0.0
        if macd_hist[i] > 0:
            momentum_score = 0.5 + 0.5 * np.clip(macd_hist[i] / (close[i] * 0.005), 0, 1)
            momentum_score += 0.3 * np.clip(macd_slope[i] / (close[i] * 0.001), 0, 1)
        else:
            momentum_score = 0.5 - 0.5 * np.clip(abs(macd_hist[i]) / (close[i] * 0.005), 0, 1)
            momentum_score -= 0.3 * np.clip(abs(macd_slope[i]) / (close[i] * 0.001), 0, 1)
        momentum_score = np.clip(momentum_score, 0, 1)
        
        # TRENDING SIGNAL
        trend_signal = 0.0
        if trending_w > 0.3:
            trend_signal = trend_strength
            if adx[i] >= ADX_STRONG_THRESHOLD:
                trend_signal *= 1.15
            trend_signal *= (0.6 + 0.4 * momentum_score)
            trend_signal *= volume_conf
        
        # RANGING SIGNAL (RSI mean reversion, trend-aligned)
        ranging_signal = 0.0
        if ranging_w > 0.3:
            rsi_signal = 0.0
            if rsi[i] < RSI_OVERSOLD:
                rsi_signal = 0.3 + 0.3 * (RSI_OVERSOLD - rsi[i]) / 35
            elif rsi[i] > RSI_OVERBOUGHT:
                rsi_signal = -0.3 - 0.3 * (rsi[i] - RSI_OVERBOUGHT) / 35
            
            # Only trade RSI extremes with trend alignment
            if rsi_signal > 0 and trend_strength > -0.2:
                ranging_signal = rsi_signal * volume_conf
            elif rsi_signal < 0 and trend_strength < 0.2:
                ranging_signal = rsi_signal * volume_conf
            else:
                ranging_signal = rsi_signal * 0.5  # Reduce if fighting trend
            
            # Extreme RSI override
            if rsi[i] < RSI_EXTREME_LOW and trend_strength > -0.3:
                ranging_signal = max(ranging_signal, 0.5 * volume_conf)
            elif rsi[i] > RSI_EXTREME_HIGH and trend_strength < 0.3:
                ranging_signal = min(ranging_signal, -0.5 * volume_conf)
        
        # BREAKOUT SIGNAL
        breakout_signal = 0.0
        is_squeeze = bb_width[i] < BB_SQUEEZE_THRESHOLD
        if breakout_w > 0.3 and is_squeeze:
            if trend_strength > 0.15 and volume_conf > 0.8:
                breakout_signal = trend_strength * (0.6 + 0.4 * breakout_w) * volume_conf
            elif trend_strength < -0.15 and volume_conf > 0.8:
                breakout_signal = trend_strength * (0.6 + 0.4 * breakout_w) * volume_conf
        
        # BLEND signals based on regime weights
        raw_signal = (
            trending_w * trend_signal +
            ranging_w * ranging_signal +
            breakout_w * breakout_signal
        )
        
        # Volatility-based position sizing
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        raw_signal *= vol_factor
        
        # Adaptive hysteresis (higher in high volatility)
        hysteresis = HYSTERESIS_BASE * (1 + atr_pct / VOLATILITY_TARGET)
        hysteresis = np.clip(hysteresis, 0.04, 0.10)
        
        # Exponential smoothing
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Apply hysteresis to reduce flipping
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < hysteresis:
                smoothed_signal = prev_signal
        
        # Apply minimum signal threshold
        regime_min = (
            trending_w * MIN_SIGNAL_TRENDING +
            ranging_w * MIN_SIGNAL_RANGING +
            breakout_w * MIN_SIGNAL_BREAKOUT
        )
        if abs(smoothed_signal) < regime_min:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals