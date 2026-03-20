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
    Building on adaptive_regime_trend_v3 success (Sharpe=0.531), improving:
    - Simplified regime detection (reduce over-fitting)
    - Stronger funding rate filter (avoid extreme funding against position)
    - Adaptive volume thresholds (percentile-based instead of fixed)
    - Better EMA stack alignment for trend confirmation
    - Improved signal smoothing with momentum-based hysteresis
    - Cleaner code structure for maintainability
    
    Key improvements over adaptive_regime_trend_v3:
    - Simplified regime logic (2 regimes instead of 3)
    - Funding rate as hard filter (not just signal modifier)
    - Volume percentile ranking for adaptive thresholds
    - Better trend strength from EMA stack alignment
    - Momentum-based hysteresis (signal change rate matters)
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

name = "adaptive_regime_trend_v4"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for better risk-adjusted returns

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
VOLUME_SPIKE_PERCENTILE = 75  # Top 25% volume

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.008
VOLATILITY_MIN = 0.002
VOLATILITY_MAX = 0.040

# Signal configuration
MIN_SIGNAL_TRENDING = 0.20
MIN_SIGNAL_RANGING = 0.30
MAX_SIGNAL = 0.70
SMOOTHING_FACTOR = 0.70
HYSTERESIS_THRESHOLD = 0.10

# Funding rate configuration
FUNDING_EXTREME_THRESHOLD = 0.0008  # 0.08% per 8h
FUNDING_FILTER_WEIGHT = 0.5  # Reduce signal by 50% if funding opposes

# Regime transition smoothing
REGIME_MEMORY = 5  # Bars to remember previous regime


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
    Returns value in [0, 1] where 1 = highest volume in lookback.
    Only uses past volume data (no look-ahead).
    """
    n = len(volume)
    volume_pct = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return volume_pct
    
    for i in range(lookback - 1, n):
        window = volume[i - lookback + 1:i + 1]
        rank = np.sum(window <= volume[i]) / lookback
        volume_pct[i] = rank
    
    return volume_pct


def calculate_funding_filter(funding_rate: np.ndarray, threshold: float = 0.0008) -> np.ndarray:
    """
    Calculate funding rate filter for position sizing.
    Extreme funding reduces position size if trading against it.
    Returns value in [0, 1] where 1 = no filter, 0 = blocked.
    Only uses current/past funding rate (no look-ahead).
    """
    n = len(funding_rate)
    filter_val = np.ones(n, dtype=np.float64)
    
    for i in range(n):
        if abs(funding_rate[i]) > threshold:
            # Extreme funding - reduce position size
            filter_val[i] = 1.0 - FUNDING_FILTER_WEIGHT
        else:
            filter_val[i] = 1.0
    
    return filter_val


def calculate_ema_stack_alignment(close: float, ema_fast: float, ema_medium: float, 
                                   ema_slow: float, ema_major: float) -> float:
    """
    Calculate EMA stack alignment score.
    Returns value in [-1, 1] where:
    - 1 = perfect bullish alignment (fast > medium > slow > major)
    - -1 = perfect bearish alignment (fast < medium < slow < major)
    - 0 = mixed/no alignment
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    # Check bullish alignment
    bullish_score = 0.0
    if ema_fast > ema_medium > ema_slow > ema_major:
        bullish_score = 1.0
    elif ema_fast > ema_medium > ema_slow:
        bullish_score = 0.7
    elif ema_fast > ema_medium:
        bullish_score = 0.4
    elif close > ema_major:
        bullish_score = 0.2
    
    # Check bearish alignment
    bearish_score = 0.0
    if ema_fast < ema_medium < ema_slow < ema_major:
        bearish_score = 1.0
    elif ema_fast < ema_medium < ema_slow:
        bearish_score = 0.7
    elif ema_fast < ema_medium:
        bearish_score = 0.4
    elif close < ema_major:
        bearish_score = 0.2
    
    # Return net alignment
    alignment = bullish_score - bearish_score
    
    # Scale by deviation from major EMA
    deviation = (close - ema_major) / ema_major
    alignment *= np.clip(abs(deviation) * 50, 0.5, 2.0)
    
    return np.clip(alignment, -1.0, 1.0)


def determine_regime(adx: float, bb_width: float, adx_trend: float, 
                     bb_squeeze: float) -> str:
    """
    Determine market regime based on ADX and Bollinger Band width.
    Returns: 'trending' or 'ranging'
    """
    is_squeeze = bb_width < bb_squeeze
    is_trending = adx >= adx_trend
    
    if is_trending and not is_squeeze:
        return 'trending'
    elif is_squeeze:
        return 'ranging'  # Squeeze suggests consolidation
    elif adx < adx_trend * 0.8:
        return 'ranging'
    else:
        return 'trending'


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Regime Trend V4 Strategy.
    
    Signal Logic:
    1. Calculate regime (trending vs ranging)
    2. Apply regime-specific logic
    3. EMA stack alignment for trend confirmation
    4. RSI for mean reversion in ranging markets
    5. Volume percentile for breakout confirmation
    6. Funding rate filter (reduce position against extreme funding)
    7. MACD histogram for momentum confirmation
    8. Signal smoothing with momentum-based hysteresis
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, funding_rate, ...]
    
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
        
        # Try to get funding rate if available
        try:
            funding_rate = prices["funding_rate"].values.astype(np.float64)
            funding_rate = np.nan_to_num(funding_rate, nan=0.0)
        except (KeyError, TypeError, ValueError):
            funding_rate = np.zeros(n, dtype=np.float64)
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
    funding_filter = calculate_funding_filter(funding_rate, FUNDING_EXTREME_THRESHOLD)
    
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
    
    # Track previous signal for smoothing and hysteresis
    prev_signal = 0.0
    prev_direction = 0  # 0=neutral, 1=long, -1=short
    prev_regime = 'ranging'
    regime_memory = ['ranging'] * REGIME_MEMORY  # Track recent regimes
    
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
        
        # Determine regime
        regime = determine_regime(adx[i], bb_width[i], ADX_TREND_THRESHOLD, BB_SQUEEZE_THRESHOLD)
        
        # Update regime memory
        regime_memory.pop(0)
        regime_memory.append(regime)
        recent_trending = sum(1 for r in regime_memory if r == 'trending') / REGIME_MEMORY
        
        # Calculate EMA stack alignment (trend strength)
        ema_alignment = calculate_ema_stack_alignment(
            close[i], ema_fast[i], ema_medium[i],
            ema_slow[i], ema_major[i]
        )
        
        # Volume confirmation (high percentile = strong volume)
        volume_confirmed = volume_pct[i] >= (VOLUME_SPIKE_PERCENTILE / 100.0)
        
        # Initialize raw signal
        raw_signal = 0.0
        
        # TRENDING REGIME
        if regime == 'trending':
            # Base signal from EMA alignment
            raw_signal = ema_alignment
            
            # MACD momentum confirmation
            macd_momentum = np.sign(macd_hist[i]) * np.clip(abs(macd_hist[i]) / (close[i] * 0.001), 0, 1)
            raw_signal *= (0.6 + 0.4 * (0.5 + 0.5 * macd_momentum))
            
            # Volume boost for trend confirmation
            if volume_confirmed:
                raw_signal *= 1.15
            
            # Apply funding filter
            raw_signal *= funding_filter[i]
            
            # Minimum signal threshold
            if abs(raw_signal) < MIN_SIGNAL_TRENDING:
                raw_signal = 0.0
        
        # RANGING REGIME
        else:
            # RSI-based mean reversion signal
            if rsi[i] < RSI_OVERSOLD:
                raw_signal = 0.5  # Long signal
            elif rsi[i] > RSI_OVERBOUGHT:
                raw_signal = -0.5  # Short signal
            else:
                raw_signal = 0.0
            
            # Amplify at extremes
            if rsi[i] < RSI_EXTREME_LOW:
                raw_signal = 0.65
            elif rsi[i] > RSI_EXTREME_HIGH:
                raw_signal = -0.65
            
            # Apply funding filter (mean reversion benefit)
            raw_signal *= funding_filter[i]
            
            # Reduce if trend is fighting mean reversion
            if abs(ema_alignment) > 0.4:
                raw_signal *= 0.5
            
            # Minimum signal threshold
            if abs(raw_signal) < MIN_SIGNAL_RANGING:
                raw_signal = 0.0
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 1.8)
        
        raw_signal *= vol_factor
        
        # Apply exponential smoothing
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Apply momentum-based hysteresis
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            # Check if signal change exceeds hysteresis threshold
            signal_change = abs(smoothed_signal - prev_signal)
            if signal_change < HYSTERESIS_THRESHOLD:
                smoothed_signal = prev_signal  # Keep previous direction
        
        # Apply funding rate directional filter
        # If funding is extreme positive, reduce long signals
        # If funding is extreme negative, reduce short signals
        if abs(funding_rate[i]) > FUNDING_EXTREME_THRESHOLD:
            if funding_rate[i] > 0 and smoothed_signal > 0:
                smoothed_signal *= 0.6  # Reduce long when funding is positive
            elif funding_rate[i] < 0 and smoothed_signal < 0:
                smoothed_signal *= 0.6  # Reduce short when funding is negative
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
        prev_regime = regime
    
    return signals