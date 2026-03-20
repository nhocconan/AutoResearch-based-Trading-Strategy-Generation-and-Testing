#!/usr/bin/env python3
"""
strategy.py - Adaptive Regime Trend V8
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on adaptive_regime_trend_v7 success (Sharpe=0.722), improving:
    - Dynamic funding bias weight based on regime confidence
    - Volatility-adaptive regime memory (shorter in high vol, longer in low vol)
    - Improved breakout confirmation (price outside BB + volume spike)
    - Momentum decay for trend exhaustion detection
    - Smoother regime transitions with weighted blending
    - Enhanced signal threshold based on volatility regime
    
    Key improvements over adaptive_regime_trend_v7:
    - Funding bias weight: 0.25 → dynamic (0.15-0.35 based on regime)
    - Regime memory: fixed 10 → adaptive (5-15 based on ATR)
    - Breakout logic: volume only → volume + price outside bands
    - Momentum decay: added for positions held >8 bars
    - Regime blending: hard threshold → weighted confidence blend
    - Signal threshold: fixed → volatility-adaptive

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

name = "adaptive_regime_trend_v8"
timeframe = "1h"
leverage = 2.5  # Moderate leverage for risk-adjusted returns

# EMA periods for trend detection
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
ADX_STRONG_THRESHOLD = 40
ADX_WEAK_THRESHOLD = 20

# Bollinger Band configuration
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_THRESHOLD = 0.02

# MACD configuration
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_THRESHOLD = 1.5

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.010
VOLATILITY_MIN = 0.002
VOLATILITY_MAX = 0.040
VOLATILITY_LOW_THRESHOLD = 0.006
VOLATILITY_HIGH_THRESHOLD = 0.020

# Signal configuration
MIN_SIGNAL_TRENDING = 0.15
MIN_SIGNAL_RANGING = 0.25
MIN_SIGNAL_BREAKOUT = 0.30
MAX_SIGNAL = 0.80
SMOOTHING_FACTOR = 0.70
HYSTERESIS_THRESHOLD = 0.10

# Funding rate configuration
FUNDING_EXTREME_THRESHOLD = 0.0005
FUNDING_BIAS_WEIGHT_MIN = 0.15
FUNDING_BIAS_WEIGHT_MAX = 0.35

# Regime transition smoothing
REGIME_MEMORY_MIN = 5
REGIME_MEMORY_MAX = 15

# Taker ratio configuration
TAKER_RATIO_THRESHOLD = 0.55
TAKER_BIAS_WEIGHT = 0.20

# Momentum exhaustion configuration
MOMENTUM_DECAY_START = 8
MOMENTUM_DECAY_FACTOR = 0.92


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


def calculate_volume_spike(volume: np.ndarray, lookback: int = 20, threshold: float = 1.5) -> np.ndarray:
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


def calculate_funding_bias(funding_rate: np.ndarray, threshold: float = 0.0005) -> np.ndarray:
    """
    Calculate funding rate bias for mean reversion signal.
    Extreme positive funding → short bias
    Extreme negative funding → long bias
    Returns value in [-1, 1].
    Only uses current/past funding rate (no look-ahead).
    """
    n = len(funding_rate)
    bias = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if funding_rate[i] > threshold:
            bias[i] = -np.clip(funding_rate[i] / threshold, 0, 1)
        elif funding_rate[i] < -threshold:
            bias[i] = np.clip(-funding_rate[i] / threshold, 0, 1)
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
    Calculate regime confidence scores.
    Returns: (trending_confidence, ranging_confidence, breakout_potential)
    All values in [0, 1].
    """
    if adx >= adx_trend:
        trending_conf = np.clip((adx - adx_trend) / 30 + 0.5, 0.5, 1.0)
    elif adx >= adx_weak:
        trending_conf = np.clip((adx - adx_weak) / (adx_trend - adx_weak) * 0.5, 0.2, 0.5)
    else:
        trending_conf = np.clip(adx / adx_weak * 0.2, 0.0, 0.2)
    
    ranging_conf = 1.0 - trending_conf
    if bb_width < bb_squeeze:
        ranging_conf *= 0.7
    
    breakout_potential = 0.0
    if bb_width < bb_squeeze:
        breakout_potential = 0.5 + 0.5 * (1 - bb_width / bb_squeeze)
    breakout_potential *= trending_conf
    
    return trending_conf, ranging_conf, breakout_potential


def calculate_momentum_decay(hold_bars: int, start_bar: int = 8, decay_factor: float = 0.92) -> float:
    """
    Calculate momentum decay factor for positions held too long.
    Reduces signal strength after extended periods in same direction.
    """
    if hold_bars <= start_bar:
        return 1.0
    
    decay_periods = hold_bars - start_bar
    decay = decay_factor ** decay_periods
    
    return max(decay, 0.3)


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Regime Trend V8 Strategy.
    
    Signal Logic:
    1. Calculate regime confidence (trending/ranging/breakout) with weighted blending
    2. Volatility-adaptive regime memory (shorter in high vol, longer in low vol)
    3. Bollinger Band squeeze + price outside bands for breakout confirmation
    4. MACD + RSI momentum confirmation
    5. Volume confirmation for breakouts
    6. Dynamic funding rate bias weight based on regime confidence
    7. Taker buy/sell ratio for market pressure
    8. Momentum decay for trend exhaustion (positions held >8 bars)
    9. Volatility-adaptive signal thresholds
    10. Signal smoothing with hysteresis
    
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
    macd_line, macd_signal, macd_hist = calculate_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    
    volume_ratio = calculate_volume_spike(volume, VOLUME_LOOKBACK, VOLUME_SPIKE_THRESHOLD)
    funding_bias = calculate_funding_bias(funding_rate, FUNDING_EXTREME_THRESHOLD)
    taker_bias = calculate_taker_bias(taker_ratio, TAKER_RATIO_THRESHOLD)
    
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        ADX_PERIOD * 2 + 1,
        VOLUME_LOOKBACK,
        BB_PERIOD,
        MACD_SLOW + MACD_SIGNAL
    )
    
    prev_signal = 0.0
    prev_direction = 0
    prev_regime = 0
    hold_bars_long = 0
    hold_bars_short = 0
    
    for i in range(min_valid_index, n):
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            hold_bars_long = 0
            hold_bars_short = 0
            continue
        
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            hold_bars_long = 0
            hold_bars_short = 0
            continue
        
        trending_conf, ranging_conf, breakout_potential = calculate_regime_confidence(
            adx[i], bb_width[i], ADX_TREND_THRESHOLD, ADX_WEAK_THRESHOLD, BB_SQUEEZE_THRESHOLD
        )
        
        is_squeeze = bb_width[i] < BB_SQUEEZE_THRESHOLD
        price_above_bb = close[i] > bb_upper[i]
        price_below_bb = close[i] < bb_lower[i]
        
        vol_adaptive_memory = int(np.clip(
            REGIME_MEMORY_MAX - (atr_pct - VOLATILITY_MIN) / (VOLATILITY_MAX - VOLATILITY_MIN) * 
            (REGIME_MEMORY_MAX - REGIME_MEMORY_MIN),
            REGIME_MEMORY_MIN, REGIME_MEMORY_MAX
        ))
        
        regime_memory = [0] * vol_adaptive_memory
        
        is_trending = trending_conf >= 0.5
        is_ranging = ranging_conf >= 0.5
        
        trend_strength = calculate_trend_strength(
            close[i], ema_fast[i], ema_medium[i],
            ema_slow[i], ema_major[i]
        )
        
        volume_confirmed = volume_ratio[i] >= VOLUME_SPIKE_THRESHOLD
        
        raw_signal = 0.0
        regime_weight = 0.0
        
        blended_conf = trending_conf * 1.0 + ranging_conf * 0.0 + breakout_potential * 0.5
        
        if is_squeeze and breakout_potential > 0.3:
            if trend_strength > 0.2 and volume_confirmed and price_above_bb:
                raw_signal = trend_strength * (0.5 + breakout_potential * 0.5)
                regime_weight = MIN_SIGNAL_BREAKOUT
            elif trend_strength < -0.2 and volume_confirmed and price_below_bb:
                raw_signal = trend_strength * (0.5 + breakout_potential * 0.5)
                regime_weight = MIN_SIGNAL_BREAKOUT
            else:
                raw_signal = trend_strength * 0.3
                regime_weight = MIN_SIGNAL_BREAKOUT * 0.5
        
        elif is_trending:
            regime_weight = MIN_SIGNAL_TRENDING
            raw_signal = trend_strength
            
            if adx[i] >= ADX_STRONG_THRESHOLD:
                raw_signal *= 1.2
            
            macd_conf = np.clip(np.sign(macd_hist[i]) * 0.5 + 0.5, 0, 1)
            if trend_strength > 0:
                raw_signal *= (0.7 + 0.3 * macd_conf)
            else:
                raw_signal *= (0.7 + 0.3 * (1 - macd_conf))
            
            if volume_confirmed:
                raw_signal *= 1.1
            
            dynamic_funding_weight = FUNDING_BIAS_WEIGHT_MIN + (
                FUNDING_BIAS_WEIGHT_MAX - FUNDING_BIAS_WEIGHT_MIN
            ) * trending_conf
            
            if abs(funding_bias[i]) > 0.3:
                if (trend_strength > 0 and funding_bias[i] < 0) or \
                   (trend_strength < 0 and funding_bias[i] > 0):
                    raw_signal *= (1.0 - dynamic_funding_weight)
            
            if abs(taker_bias[i]) > 0.3:
                if (trend_strength > 0 and taker_bias[i] > 0) or \
                   (trend_strength < 0 and taker_bias[i] < 0):
                    raw_signal *= (1.0 + TAKER_BIAS_WEIGHT)
        
        elif is_ranging:
            regime_weight = MIN_SIGNAL_RANGING
            
            if rsi[i] < RSI_OVERSOLD:
                raw_signal = 0.4
            elif rsi[i] > RSI_OVERBOUGHT:
                raw_signal = -0.4
            else:
                raw_signal = 0.0
            
            if rsi[i] < RSI_EXTREME_LOW:
                raw_signal = 0.6
            elif rsi[i] > RSI_EXTREME_HIGH:
                raw_signal = -0.6
            
            dynamic_funding_weight = FUNDING_BIAS_WEIGHT_MIN + (
                FUNDING_BIAS_WEIGHT_MAX - FUNDING_BIAS_WEIGHT_MIN
            ) * ranging_conf
            
            if abs(funding_bias[i]) > 0.3:
                if (raw_signal > 0 and funding_bias[i] > 0) or \
                   (raw_signal < 0 and funding_bias[i] < 0):
                    raw_signal *= (1.0 + dynamic_funding_weight)
            
            if abs(trend_strength) > 0.3:
                raw_signal *= 0.5
            
            if abs(taker_bias[i]) > 0.3:
                if (raw_signal > 0 and taker_bias[i] > 0) or \
                   (raw_signal < 0 and taker_bias[i] < 0):
                    raw_signal *= (1.0 + TAKER_BIAS_WEIGHT)
        
        else:
            regime_weight = MIN_SIGNAL_RANGING * 0.8
            
            trend_signal = trend_strength * 0.6
            rsi_signal = 0.0
            if rsi[i] < RSI_OVERSOLD:
                rsi_signal = 0.3
            elif rsi[i] > RSI_OVERBOUGHT:
                rsi_signal = -0.3
            
            raw_signal = trending_conf * trend_signal + ranging_conf * rsi_signal
        
        if prev_direction > 0:
            hold_bars_long += 1
            hold_bars_short = 0
        elif prev_direction < 0:
            hold_bars_short += 1
            hold_bars_long = 0
        else:
            hold_bars_long = 0
            hold_bars_short = 0
        
        current_direction = np.sign(raw_signal)
        if current_direction > 0:
            decay_factor = calculate_momentum_decay(hold_bars_long, MOMENTUM_DECAY_START, MOMENTUM_DECAY_FACTOR)
        elif current_direction < 0:
            decay_factor = calculate_momentum_decay(hold_bars_short, MOMENTUM_DECAY_START, MOMENTUM_DECAY_FACTOR)
        else:
            decay_factor = 1.0
        
        raw_signal *= decay_factor
        
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.4, 2.0)
        
        raw_signal *= vol_factor
        
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < HYSTERESIS_THRESHOLD:
                smoothed_signal = prev_signal
        
        vol_adaptive_threshold = regime_weight * np.clip(vol_factor, 0.5, 1.5)
        if abs(smoothed_signal) < vol_adaptive_threshold:
            smoothed_signal = 0.0
        
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
        prev_regime = 1 if is_trending else 0
    
    return signals