#!/usr/bin/env python3
"""
strategy.py - Adaptive Regime Trend V4
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on adaptive_regime_trend_v2 success (Sharpe=0.500), improving:
    - Funding rate bias: extreme funding → contrarian signal
    - Simplified momentum calculation (reduce noise from v3)
    - Better regime transition smoothing (exponential decay)
    - Volatility-adaptive entry thresholds
    - Cleaner signal combination logic
    
    Key improvements over adaptive_regime_trend_v2:
    - Funding rate awareness (if available in prices DataFrame)
    - Simplified momentum score (RSI + price momentum only)
    - Exponential regime memory (recent bars weighted more)
    - Dynamic signal thresholds based on volatility
    - Reduced parameter complexity

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

name = "adaptive_regime_trend_v4"
timeframe = "1h"
leverage = 2.5  # Conservative leverage for crypto futures

# EMA periods for trend detection
EMA_FAST = 9
EMA_MEDIUM = 21
EMA_SLOW = 50
EMA_MAJOR = 200

# RSI configuration
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
RSI_EXTREME_HIGH = 80
RSI_EXTREME_LOW = 20

# ADX regime detection
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_STRONG_THRESHOLD = 40
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
VOLUME_PERCENTILE_THRESHOLD = 0.65

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012
VOLATILITY_MIN = 0.003
VOLATILITY_MAX = 0.060

# Signal configuration
MIN_SIGNAL_TRENDING = 0.15
MIN_SIGNAL_RANGING = 0.25
MIN_SIGNAL_BREAKOUT = 0.30
MAX_SIGNAL = 0.85
SMOOTHING_FACTOR = 0.75
HYSTERESIS_THRESHOLD = 0.08

# Funding rate configuration
FUNDING_EXTREME_THRESHOLD = 0.0008  # 0.08% per 8h = extreme
FUNDING_BIAS_STRENGTH = 0.3  # How much funding affects signal

# Regime transition smoothing
REGIME_DECAY = 0.85  # Exponential decay for regime memory


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


def calculate_price_momentum(close: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Calculate simple price momentum (rate of change).
    Returns value normalized to [-1, 1].
    """
    n = len(close)
    momentum = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return momentum
    
    for i in range(lookback, n):
        if close[i-lookback] > 0:
            roc = (close[i] - close[i-lookback]) / close[i-lookback]
            # Normalize: assume max 5% move in lookback period
            momentum[i] = np.clip(roc / 0.05, -1, 1)
    
    return momentum


def get_funding_bias(prices: pd.DataFrame, i: int) -> float:
    """
    Extract funding rate bias from prices DataFrame if available.
    Returns value in [-1, 1] where positive = bullish funding (contrarian short signal).
    Only uses past funding data (no look-ahead).
    """
    try:
        if 'funding_rate' not in prices.columns:
            return 0.0
        
        funding = prices['funding_rate'].values
        if i < 1 or np.isnan(funding[i]):
            return 0.0
        
        # Use current funding rate (already known at bar close)
        funding_rate = funding[i]
        
        # Extreme funding suggests crowded position → contrarian signal
        if funding_rate > FUNDING_EXTREME_THRESHOLD:
            return -FUNDING_BIAS_STRENGTH  # Extreme long funding → short bias
        elif funding_rate < -FUNDING_EXTREME_THRESHOLD:
            return FUNDING_BIAS_STRENGTH  # Extreme short funding → long bias
        else:
            # Linear scaling for moderate funding
            return -funding_rate * (FUNDING_BIAS_STRENGTH / FUNDING_EXTREME_THRESHOLD)
    except (KeyError, TypeError, IndexError):
        return 0.0


def calculate_trend_strength(close: float, ema_fast: float, ema_medium: float, 
                             ema_slow: float, ema_major: float) -> float:
    """
    Calculate trend strength score based on EMA stack alignment.
    Returns value in [-1, 1] where magnitude indicates strength.
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    # Calculate normalized deviations
    fast_dev = (ema_fast - ema_medium) / close
    medium_dev = (ema_medium - ema_slow) / close
    slow_dev = (ema_slow - ema_major) / close
    major_dev = (close - ema_major) / close
    
    # Direction from major trend
    major_direction = np.sign(major_dev)
    
    # Alignment score: all EMAs should be in same order
    if major_direction > 0:
        # Bullish alignment: fast > medium > slow > major
        alignment = 0.0
        if ema_fast > ema_medium:
            alignment += 0.4
        if ema_medium > ema_slow:
            alignment += 0.35
        if ema_slow > ema_major:
            alignment += 0.25
        trend_strength = alignment * major_direction
    else:
        # Bearish alignment: fast < medium < slow < major
        alignment = 0.0
        if ema_fast < ema_medium:
            alignment += 0.4
        if ema_medium < ema_slow:
            alignment += 0.35
        if ema_slow < ema_major:
            alignment += 0.25
        trend_strength = -alignment
    
    # Scale by magnitude of deviations
    avg_dev = abs(fast_dev + medium_dev + slow_dev) / 3
    trend_strength *= np.clip(avg_dev * 100, 0.5, 2.0)
    
    return np.clip(trend_strength, -1.0, 1.0)


def calculate_regime_confidence(adx: float, bb_width: float) -> tuple:
    """
    Calculate regime confidence scores.
    Returns: (trending_confidence, ranging_confidence, breakout_potential)
    All values in [0, 1].
    """
    # Trending confidence from ADX
    if adx >= ADX_TREND_THRESHOLD:
        trending_conf = np.clip((adx - ADX_TREND_THRESHOLD) / 30 + 0.5, 0.5, 1.0)
    elif adx >= ADX_WEAK_THRESHOLD:
        trending_conf = np.clip((adx - ADX_WEAK_THRESHOLD) / (ADX_TREND_THRESHOLD - ADX_WEAK_THRESHOLD) * 0.5, 0.2, 0.5)
    else:
        trending_conf = np.clip(adx / ADX_WEAK_THRESHOLD * 0.2, 0.0, 0.2)
    
    # Ranging confidence (inverse of trending)
    ranging_conf = 1.0 - trending_conf
    
    # Breakout potential (high when squeeze)
    breakout_potential = 0.0
    if bb_width < BB_SQUEEZE_THRESHOLD:
        breakout_potential = 0.5 + 0.5 * (1 - bb_width / BB_SQUEEZE_THRESHOLD)
    
    return trending_conf, ranging_conf, breakout_potential


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Regime Trend V4 Strategy.
    
    Signal Logic:
    1. Calculate regime confidence (trending/ranging/breakout)
    2. Apply logic weighted by regime confidence
    3. Bollinger Band squeeze detection for breakout preparation
    4. RSI + price momentum confirmation (simplified from v3)
    5. Volume confirmation for breakouts
    6. Funding rate bias (contrarian at extremes)
    7. Volatility-adaptive position sizing
    8. Signal smoothing with exponential regime memory
    
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
    price_momentum = calculate_price_momentum(close, lookback=5)
    
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
    
    # Track previous signal for smoothing
    prev_signal = 0.0
    prev_direction = 0  # 0=neutral, 1=long, -1=short
    
    # Exponential regime memory (recent bars weighted more)
    regime_memory = 0.5  # Start neutral
    
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
            adx[i], bb_width[i]
        )
        
        # Update exponential regime memory
        current_regime = 1.0 if trending_conf > 0.5 else 0.0
        regime_memory = REGIME_DECAY * regime_memory + (1 - REGIME_DECAY) * current_regime
        
        # Determine dominant regime
        is_squeeze = bb_width[i] < BB_SQUEEZE_THRESHOLD
        is_trending = trending_conf >= 0.5 and regime_memory >= 0.4
        is_ranging = ranging_conf >= 0.5 and regime_memory <= 0.6
        
        # Calculate trend strength
        trend_strength = calculate_trend_strength(
            close[i], ema_fast[i], ema_medium[i],
            ema_slow[i], ema_major[i]
        )
        
        # Volume confirmation
        volume_confirmed = volume_pct[i] >= VOLUME_PERCENTILE_THRESHOLD
        
        # Get funding rate bias
        funding_bias = get_funding_bias(prices, i)
        
        # Initialize raw signal and regime weight
        raw_signal = 0.0
        regime_weight = 0.0
        
        # BREAKOUT REGIME (squeeze + potential expansion)
        if is_squeeze and breakout_potential > 0.3:
            # Wait for directional confirmation
            if trend_strength > 0.15 and volume_confirmed:
                raw_signal = trend_strength * (0.5 + breakout_potential * 0.5)
                regime_weight = MIN_SIGNAL_BREAKOUT
            elif trend_strength < -0.15 and volume_confirmed:
                raw_signal = trend_strength * (0.5 + breakout_potential * 0.5)
                regime_weight = MIN_SIGNAL_BREAKOUT
            else:
                # Squeeze without direction = reduce position
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
            
            # Momentum confirmation (simplified: price momentum + RSI position)
            momentum_conf = 0.5 + 0.25 * price_momentum[i] + 0.25 * (rsi[i] - 50) / 50
            momentum_conf = np.clip(momentum_conf, 0.3, 1.0)
            
            if trend_strength > 0:
                raw_signal *= momentum_conf
            else:
                raw_signal *= (1.3 - momentum_conf)
            
            # Volume boost for trend confirmation
            if volume_confirmed:
                raw_signal *= 1.1
            
            # Apply funding bias (contrarian at extremes)
            raw_signal += funding_bias * 0.3
        
        # RANGING REGIME
        elif is_ranging:
            regime_weight = MIN_SIGNAL_RANGING
            
            # RSI-based mean reversion signal
            if rsi[i] < RSI_OVERSOLD:
                raw_signal = 0.35 + 0.25 * (RSI_OVERSOLD - rsi[i]) / RSI_OVERSOLD
            elif rsi[i] > RSI_OVERBOUGHT:
                raw_signal = -0.35 - 0.25 * (rsi[i] - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT)
            else:
                raw_signal = 0.0
            
            # Amplify at extremes
            if rsi[i] < RSI_EXTREME_LOW:
                raw_signal = max(raw_signal, 0.55)
            elif rsi[i] > RSI_EXTREME_HIGH:
                raw_signal = min(raw_signal, -0.55)
            
            # Reduce signal if trend is fighting mean reversion
            if abs(trend_strength) > 0.25:
                raw_signal *= 0.6
            
            # Apply funding bias
            raw_signal += funding_bias * 0.4
        
        # TRANSITION REGIME (mixed signals)
        else:
            # Blend trending and ranging logic based on regime memory
            regime_weight = MIN_SIGNAL_RANGING * 0.9
            
            # Weighted combination
            trend_signal = trend_strength * 0.5
            rsi_signal = 0.0
            if rsi[i] < RSI_OVERSOLD:
                rsi_signal = 0.25
            elif rsi[i] > RSI_OVERBOUGHT:
                rsi_signal = -0.25
            
            raw_signal = regime_memory * trend_signal + (1 - regime_memory) * rsi_signal
            raw_signal += funding_bias * 0.2
        
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