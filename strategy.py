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
    - Funding rate mean reversion: extreme funding → contrarian signal
    - Enhanced regime confidence with weighted multi-factor scoring
    - Volume momentum (rate of change) in addition to percentile
    - Smoother regime transitions with exponential memory decay
    - More conservative signal generation to reduce whipsaws
    - Better volatility-adaptive position sizing
    
    Key improvements over adaptive_regime_trend_v2:
    - Funding rate bias: extreme positive funding → reduce long / increase short bias
    - Volume momentum: accelerating volume confirms breakouts
    - Regime confidence: multi-factor weighted score (ADX + BB + Volatility)
    - Signal smoothing: adaptive smoothing based on regime stability
    - Transition hysteresis: require stronger signal to flip direction

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
leverage = 2.5  # Conservative leverage for crypto perps

# EMA periods for trend detection
EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration
RSI_PERIOD = 14
RSI_OVERBOUGHT = 67
RSI_OVERSOLD = 33
RSI_EXTREME_HIGH = 75
RSI_EXTREME_LOW = 25

# ADX regime detection
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 23
ADX_STRONG_THRESHOLD = 35
ADX_WEAK_THRESHOLD = 17

# Bollinger Band configuration
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_THRESHOLD = 0.018

# MACD configuration
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_PERCENTILE_THRESHOLD = 0.65
VOLUME_MOMENTUM_LOOKBACK = 5

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012
VOLATILITY_MIN = 0.0025
VOLATILITY_MAX = 0.045

# Funding rate configuration (if available)
FUNDING_EXTREME_THRESHOLD = 0.0008  # 0.08% per 8hr = extreme
FUNDING_BIAS_STRENGTH = 0.3  # How much funding affects signal

# Signal configuration
MIN_SIGNAL_TRENDING = 0.15
MIN_SIGNAL_RANGING = 0.22
MIN_SIGNAL_BREAKOUT = 0.28
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR_BASE = 0.65
SMOOTHING_FACTOR_TRENDING = 0.75  # More smoothing in trends
HYSTERESIS_THRESHOLD = 0.08

# Regime transition smoothing
REGIME_MEMORY = 7  # Bars to remember previous regime
REGIME_MEMORY_DECAY = 0.7  # Exponential decay for older memories


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


def calculate_volume_percentile(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume percentile rank using rolling window.
    Only uses past volume data (no look-ahead).
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


def calculate_volume_momentum(volume: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Calculate volume momentum (rate of change) using only past data.
    Returns normalized value in approximately [-1, 1].
    """
    n = len(volume)
    vol_momentum = np.zeros(n, dtype=np.float64)
    
    if n < lookback + 1:
        return vol_momentum
    
    for i in range(lookback + 1, n):
        if volume[i-lookback] > 0:
            vol_momentum[i] = (volume[i] - volume[i-lookback]) / volume[i-lookback]
        else:
            vol_momentum[i] = 0.0
    
    # Normalize to approximately [-1, 1]
    vol_momentum = np.clip(vol_momentum / 2.0, -1.0, 1.0)
    
    return vol_momentum


def calculate_funding_bias(prices: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """
    Calculate funding rate bias signal.
    Extreme positive funding → short bias (crowded longs)
    Extreme negative funding → long bias (crowded shorts)
    
    Returns value in [-1, 1] where positive = long bias, negative = short bias
    """
    n = len(prices)
    funding_bias = np.zeros(n, dtype=np.float64)
    
    # Try to get funding rate from prices DataFrame
    if 'funding_rate' not in prices.columns:
        return funding_bias
    
    funding = prices['funding_rate'].values.astype(np.float64)
    funding = np.nan_to_num(funding, nan=0.0)
    
    if n < lookback:
        return funding_bias
    
    # Calculate rolling mean and std of funding rate
    funding_series = pd.Series(funding)
    funding_mean = funding_series.rolling(window=lookback, min_periods=lookback).mean()
    funding_std = funding_series.rolling(window=lookback, min_periods=lookback).std()
    
    for i in range(lookback, n):
        if funding_std.iloc[i] > 0:
            # Z-score of current funding rate
            z_score = (funding[i] - funding_mean.iloc[i]) / funding_std.iloc[i]
            # Invert: high funding → short bias, low funding → long bias
            funding_bias[i] = -np.clip(z_score / 3.0, -1.0, 1.0)
        else:
            # No volatility, use absolute level
            if abs(funding[i]) > FUNDING_EXTREME_THRESHOLD:
                funding_bias[i] = -np.sign(funding[i]) * 0.5
    
    return funding_bias


def calculate_momentum_score(rsi: np.ndarray, macd_hist: np.ndarray, close: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Calculate combined momentum score from RSI slope and MACD histogram.
    Returns value in [-1, 1].
    """
    n = len(close)
    momentum = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return momentum
    
    for i in range(lookback, n):
        rsi_slope = (rsi[i] - rsi[i-lookback]) / lookback if lookback > 0 else 0
        rsi_score = np.clip(rsi_slope / 10, -1, 1)
        
        macd_score = np.clip(macd_hist[i] / (close[i] * 0.01), -1, 1) if close[i] > 0 else 0
        
        momentum[i] = 0.6 * rsi_score + 0.4 * macd_score
    
    return momentum


def detect_rsi_divergence(close: np.ndarray, rsi: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Detect RSI divergence using only past data.
    Returns: 1 for bullish divergence, -1 for bearish, 0 for none
    """
    n = len(close)
    divergence = np.zeros(n, dtype=np.float64)
    
    if n < lookback * 2:
        return divergence
    
    for i in range(lookback * 2, n):
        price_low_recent = np.min(close[i-lookback:i])
        price_low_prev = np.min(close[i-lookback*2:i-lookback])
        rsi_low_recent = np.min(rsi[i-lookback:i])
        rsi_low_prev = np.min(rsi[i-lookback*2:i-lookback])
        
        if price_low_recent < price_low_prev * 0.995 and rsi_low_recent > rsi_low_prev * 1.02:
            divergence[i] = 1.0
            continue
        
        price_high_recent = np.max(close[i-lookback:i])
        price_high_prev = np.max(close[i-lookback*2:i-lookback])
        rsi_high_recent = np.max(rsi[i-lookback:i])
        rsi_high_prev = np.max(rsi[i-lookback*2:i-lookback])
        
        if price_high_recent > price_high_prev * 1.005 and rsi_high_recent < rsi_high_prev * 0.98:
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


def calculate_regime_confidence(adx: float, bb_width: float, atr_pct: float,
                                 adx_trend: float, adx_weak: float, 
                                 bb_squeeze: float, vol_target: float) -> tuple:
    """
    Calculate regime confidence scores using multi-factor weighted approach.
    Returns: (trending_confidence, ranging_confidence, breakout_potential)
    All values in [0, 1].
    """
    # ADX component (0.4 weight)
    if adx >= adx_trend:
        adx_score = np.clip((adx - adx_trend) / 30 + 0.6, 0.6, 1.0)
    elif adx >= adx_weak:
        adx_score = np.clip((adx - adx_weak) / (adx_trend - adx_weak) * 0.4 + 0.2, 0.2, 0.6)
    else:
        adx_score = np.clip(adx / adx_weak * 0.2, 0.0, 0.2)
    
    # BB width component (0.3 weight) - narrow bands suggest potential breakout
    if bb_width < bb_squeeze:
        bb_score = 0.7 + 0.3 * (1 - bb_width / bb_squeeze)  # High breakout potential
    else:
        bb_score = 0.3 + 0.4 * (1 - min(bb_width / (bb_squeeze * 3), 1.0))  # Lower confidence
    
    # Volatility component (0.3 weight) - moderate volatility preferred
    if vol_target * 0.5 <= atr_pct <= vol_target * 2.0:
        vol_score = 0.8
    elif atr_pct < vol_target * 0.5:
        vol_score = 0.4 + 0.4 * (atr_pct / (vol_target * 0.5))
    else:
        vol_score = 0.8 - 0.4 * min((atr_pct - vol_target * 2.0) / (vol_target * 2.0), 1.0)
    
    # Weighted combination
    trending_conf = 0.4 * adx_score + 0.3 * bb_score + 0.3 * vol_score
    
    # Ranging confidence (inverse relationship but not exact)
    ranging_conf = 1.0 - trending_conf * 0.8
    
    # Breakout potential (high when squeeze + rising ADX + moderate vol)
    breakout_potential = 0.0
    if bb_width < bb_squeeze:
        breakout_potential = 0.5 + 0.3 * (1 - bb_width / bb_squeeze)
        breakout_potential += 0.2 * min(adx / adx_trend, 1.0) if adx_trend > 0 else 0
    
    breakout_potential = np.clip(breakout_potential, 0.0, 1.0)
    
    return trending_conf, ranging_conf, breakout_potential


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Regime Trend V3 Strategy.
    
    Signal Logic:
    1. Calculate multi-factor regime confidence (ADX + BB + Volatility)
    2. Funding rate bias for contrarian signals at extremes
    3. Volume momentum confirmation for breakouts
    4. Adaptive signal smoothing based on regime stability
    5. Enhanced hysteresis to reduce whipsaws
    6. Volatility-adaptive position sizing
    
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
    
    volume_pct = calculate_volume_percentile(volume, VOLUME_LOOKBACK)
    vol_momentum = calculate_volume_momentum(volume, VOLUME_MOMENTUM_LOOKBACK)
    rsi_divergence = detect_rsi_divergence(close, rsi, lookback=5)
    momentum_score = calculate_momentum_score(rsi, macd_hist, close, lookback=5)
    funding_bias = calculate_funding_bias(prices, lookback=20)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        ADX_PERIOD * 2 + 1,
        VOLUME_LOOKBACK,
        VOLUME_MOMENTUM_LOOKBACK + 1,
        BB_PERIOD,
        MACD_SLOW + MACD_SIGNAL,
        20  # Funding rate lookback
    )
    
    # Track state for smoothing and hysteresis
    prev_signal = 0.0
    prev_direction = 0  # 0=neutral, 1=long, -1=short
    regime_memory = []  # Track recent regime confidence scores
    
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
            adx[i], bb_width[i], atr_pct,
            ADX_TREND_THRESHOLD, ADX_WEAK_THRESHOLD, 
            BB_SQUEEZE_THRESHOLD, VOLATILITY_TARGET
        )
        
        # Update regime memory with exponential decay
        regime_memory.append(trending_conf)
        if len(regime_memory) > REGIME_MEMORY:
            regime_memory.pop(0)
        
        # Calculate weighted regime memory (recent bars weighted more)
        if len(regime_memory) > 0:
            weights = [REGIME_MEMORY_DECAY ** j for j in range(len(regime_memory)-1, -1, -1)]
            recent_trending = sum(r * w for r, w in zip(regime_memory, weights)) / sum(weights)
        else:
            recent_trending = 0.5
        
        # Determine dominant regime
        is_squeeze = bb_width[i] < BB_SQUEEZE_THRESHOLD
        is_trending = trending_conf >= 0.55 and recent_trending >= 0.5
        is_ranging = ranging_conf >= 0.55 and recent_trending < 0.5
        
        # Calculate trend strength
        trend_strength = calculate_trend_strength(
            close[i], ema_fast[i], ema_medium[i],
            ema_slow[i], ema_major[i]
        )
        
        # Volume confirmation
        volume_confirmed = volume_pct[i] >= VOLUME_PERCENTILE_THRESHOLD
        volume_accelerating = vol_momentum[i] > 0.2
        
        # Initialize raw signal and regime weight
        raw_signal = 0.0
        regime_weight = 0.0
        
        # BREAKOUT REGIME (squeeze + potential expansion)
        if is_squeeze and breakout_potential > 0.4:
            regime_weight = MIN_SIGNAL_BREAKOUT
            
            # Require both trend strength AND volume confirmation
            if trend_strength > 0.25 and volume_confirmed and volume_accelerating:
                raw_signal = trend_strength * (0.6 + breakout_potential * 0.4)
            elif trend_strength < -0.25 and volume_confirmed and volume_accelerating:
                raw_signal = trend_strength * (0.6 + breakout_potential * 0.4)
            else:
                # Wait for confirmation
                raw_signal = 0.0
                regime_weight = 0.0
        
        # TRENDING REGIME
        elif is_trending:
            regime_weight = MIN_SIGNAL_TRENDING
            
            # Base signal from trend strength
            raw_signal = trend_strength
            
            # Amplify in strong trends
            if adx[i] >= ADX_STRONG_THRESHOLD:
                raw_signal *= 1.15
            
            # Momentum confirmation
            momentum_conf = np.clip(momentum_score[i] + 0.5, 0, 1)
            if trend_strength > 0:
                raw_signal *= (0.75 + 0.25 * momentum_conf)
            else:
                raw_signal *= (0.75 + 0.25 * (1 - momentum_conf))
            
            # Volume boost for trend confirmation
            if volume_confirmed:
                raw_signal *= 1.08
            
            # RSI divergence override (reversal signal)
            if rsi_divergence[i] != 0:
                raw_signal *= 0.5
        
        # RANGING REGIME
        elif is_ranging:
            regime_weight = MIN_SIGNAL_RANGING
            
            # RSI-based mean reversion signal
            if rsi[i] < RSI_OVERSOLD:
                raw_signal = 0.45
            elif rsi[i] > RSI_OVERBOUGHT:
                raw_signal = -0.45
            else:
                raw_signal = 0.0
            
            # Amplify at extremes
            if rsi[i] < RSI_EXTREME_LOW:
                raw_signal = 0.65
            elif rsi[i] > RSI_EXTREME_HIGH:
                raw_signal = -0.65
            
            # RSI divergence confirmation
            if rsi_divergence[i] == 1.0:
                raw_signal = max(raw_signal, 0.5)
            elif rsi_divergence[i] == -1.0:
                raw_signal = min(raw_signal, -0.5)
            
            # Reduce signal if strong trend fighting mean reversion
            if abs(trend_strength) > 0.35:
                raw_signal *= 0.4
        
        # TRANSITION REGIME (mixed signals)
        else:
            regime_weight = MIN_SIGNAL_RANGING * 0.7
            
            # Conservative blend
            trend_signal = trend_strength * 0.5
            rsi_signal = 0.0
            if rsi[i] < RSI_OVERSOLD:
                rsi_signal = 0.35
            elif rsi[i] > RSI_OVERBOUGHT:
                rsi_signal = -0.35
            
            raw_signal = trending_conf * trend_signal + ranging_conf * rsi_signal
        
        # Apply funding rate bias (contrarian signal)
        if abs(funding_bias[i]) > 0.2:
            # Funding bias acts as a dampener or amplifier
            if np.sign(raw_signal) == np.sign(funding_bias[i]):
                # Agreement: amplify
                raw_signal *= (1.0 + FUNDING_BIAS_STRENGTH * abs(funding_bias[i]))
            else:
                # Disagreement: reduce
                raw_signal *= (1.0 - FUNDING_BIAS_STRENGTH * abs(funding_bias[i]))
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        
        raw_signal *= vol_factor
        
        # Adaptive smoothing based on regime stability
        if is_trending and len(regime_memory) >= 3:
            # More stable regime → more smoothing
            regime_stability = 1.0 - np.std(regime_memory[-3:])
            smoothing_factor = SMOOTHING_FACTOR_TRENDING + 0.1 * regime_stability
        else:
            smoothing_factor = SMOOTHING_FACTOR_BASE
        
        smoothing_factor = np.clip(smoothing_factor, 0.5, 0.85)
        
        # Apply exponential smoothing
        smoothed_signal = smoothing_factor * prev_signal + (1.0 - smoothing_factor) * raw_signal
        
        # Apply hysteresis to reduce flipping
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            # Require stronger signal to flip direction
            if abs(smoothed_signal - prev_signal) < HYSTERESIS_THRESHOLD:
                smoothed_signal = prev_signal
        
        # Apply minimum signal threshold based on regime
        if abs(smoothed_signal) < regime_weight:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals