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
    - Simplified regime detection (reduce overfitting from v3-v8 failures)
    - Better volatility-adaptive position sizing
    - Cleaner trend confirmation with EMA stack
    - Improved RSI mean-reversion in ranging markets
    - Reduced parameter count for better generalization
    - More robust signal smoothing with adaptive hysteresis
    
    Key improvements over adaptive_regime_trend_v2:
    - Simplified regime logic (fewer thresholds to tune)
    - Volatility-targeted position sizing (more consistent risk)
    - Better EMA stack alignment scoring
    - Reduced signal flipping with adaptive hysteresis
    - Cleaner code structure for maintainability

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

# EMA periods for trend detection (simplified from v2)
EMA_FAST = 8
EMA_MEDIUM = 21
EMA_SLOW = 55
EMA_MAJOR = 200

# RSI configuration (slightly adjusted thresholds)
RSI_PERIOD = 14
RSI_OVERBOUGHT = 65
RSI_OVERSOLD = 35
RSI_EXTREME_HIGH = 75
RSI_EXTREME_LOW = 25

# ADX regime detection (simplified thresholds)
ADX_PERIOD = 14
ADX_TREND_THRESHOLD = 25
ADX_WEAK_THRESHOLD = 20

# Bollinger Band configuration
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_THRESHOLD = 0.012  # Slightly tighter squeeze detection

# MACD configuration
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_PERCENTILE_THRESHOLD = 0.65

# Volatility configuration (target-based sizing)
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.008  # Target ATR % for position sizing
VOLATILITY_MIN = 0.002
VOLATILITY_MAX = 0.040

# Signal configuration (simplified)
MIN_SIGNAL_STRENGTH = 0.15
MAX_SIGNAL = 0.75
SMOOTHING_FACTOR = 0.65
HYSTERESIS_BASE = 0.05

# Regime transition smoothing
REGIME_MEMORY = 3  # Reduced from 5 for faster adaptation


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


def calculate_ema_stack_score(ema_fast: float, ema_medium: float, 
                               ema_slow: float, ema_major: float, 
                               close: float) -> float:
    """
    Calculate EMA stack alignment score.
    Returns value in [-1, 1] where magnitude indicates trend strength.
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    # Check bullish alignment: close > fast > medium > slow > major
    bullish = (close > ema_fast > ema_medium > ema_slow > ema_major)
    bearish = (close < ema_fast < ema_medium < ema_slow < ema_major)
    
    if bullish:
        # Calculate alignment strength
        spread1 = (ema_fast - ema_medium) / close
        spread2 = (ema_medium - ema_slow) / close
        spread3 = (ema_slow - ema_major) / close
        avg_spread = (spread1 + spread2 + spread3) / 3
        score = 0.5 + np.clip(avg_spread * 50, 0, 0.5)
    elif bearish:
        spread1 = (ema_medium - ema_fast) / close
        spread2 = (ema_slow - ema_medium) / close
        spread3 = (ema_major - ema_slow) / close
        avg_spread = (spread1 + spread2 + spread3) / 3
        score = -0.5 - np.clip(avg_spread * 50, 0, 0.5)
    else:
        # Mixed alignment - use major trend direction
        major_trend = np.sign(close - ema_major)
        # Partial alignment score
        alignment_count = 0
        if (close - ema_major) * (ema_fast - ema_medium) > 0:
            alignment_count += 1
        if (close - ema_major) * (ema_medium - ema_slow) > 0:
            alignment_count += 1
        score = major_trend * (0.2 + 0.1 * alignment_count)
    
    return np.clip(score, -1.0, 1.0)


def detect_regime(adx: float, bb_width: float, adx_trend: float, 
                  adx_weak: float, bb_squeeze: float) -> str:
    """
    Detect market regime based on ADX and Bollinger Band width.
    Returns: 'trending', 'ranging', 'squeeze', or 'transition'
    """
    is_squeeze = bb_width < bb_squeeze
    
    if is_squeeze and adx < adx_weak:
        return 'squeeze'
    elif adx >= adx_trend:
        return 'trending'
    elif adx < adx_weak:
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
    1. Detect market regime (trending/ranging/squeeze/transition)
    2. Apply regime-specific signal generation
    3. Use EMA stack for trend confirmation
    4. RSI for mean-reversion in ranging markets
    5. Volume confirmation for breakouts
    6. Volatility-targeted position sizing
    7. Signal smoothing with adaptive hysteresis
    
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
    
    # Track state for smoothing
    prev_signal = 0.0
    prev_direction = 0  # 0=neutral, 1=long, -1=short
    regime_memory = [0] * REGIME_MEMORY  # 0=ranging, 1=trending
    
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
        
        # Detect current regime
        regime = detect_regime(
            adx[i], bb_width[i], 
            ADX_TREND_THRESHOLD, ADX_WEAK_THRESHOLD, 
            BB_SQUEEZE_THRESHOLD
        )
        
        # Update regime memory
        regime_memory.pop(0)
        regime_memory.append(1 if regime == 'trending' else 0)
        recent_trending = sum(regime_memory) / REGIME_MEMORY
        
        # Calculate EMA stack score (trend strength and direction)
        ema_score = calculate_ema_stack_score(
            ema_fast[i], ema_medium[i], 
            ema_slow[i], ema_major[i], 
            close[i]
        )
        
        # Volume confirmation
        volume_confirmed = volume_pct[i] >= VOLUME_PERCENTILE_THRESHOLD
        
        # Initialize raw signal based on regime
        raw_signal = 0.0
        
        # TRENDING REGIME
        if regime == 'trending':
            # Follow the trend with EMA stack confirmation
            raw_signal = ema_score
            
            # Amplify strong trends
            if adx[i] > 35:
                raw_signal *= 1.15
            
            # Volume boost for confirmation
            if volume_confirmed:
                raw_signal *= 1.1
            
            # MACD confirmation
            macd_conf = np.sign(macd_hist[i])
            if np.sign(raw_signal) == macd_conf:
                raw_signal *= 1.05
        
        # RANGING REGIME
        elif regime == 'ranging':
            # Mean reversion with RSI
            if rsi[i] < RSI_OVERSOLD:
                raw_signal = 0.35
            elif rsi[i] > RSI_OVERBOUGHT:
                raw_signal = -0.35
            else:
                raw_signal = 0.0
            
            # Amplify at extremes
            if rsi[i] < RSI_EXTREME_LOW:
                raw_signal = 0.5
            elif rsi[i] > RSI_EXTREME_HIGH:
                raw_signal = -0.5
            
            # Reduce if fighting major trend
            if abs(ema_score) > 0.3 and np.sign(raw_signal) != np.sign(ema_score):
                raw_signal *= 0.5
        
        # SQUEEZE REGIME (low volatility, potential breakout)
        elif regime == 'squeeze':
            # Wait for directional bias from EMA stack
            if abs(ema_score) > 0.15:
                raw_signal = ema_score * 0.6
                # Require volume confirmation for breakout
                if not volume_confirmed:
                    raw_signal *= 0.7
            else:
                raw_signal = 0.0
        
        # TRANSITION REGIME
        else:  # transition
            # Blend trend and mean-reversion signals
            trend_component = ema_score * 0.5
            rsi_component = 0.0
            if rsi[i] < RSI_OVERSOLD:
                rsi_component = 0.2
            elif rsi[i] > RSI_OVERBOUGHT:
                rsi_component = -0.2
            
            # Weight by recent regime tendency
            raw_signal = recent_trending * trend_component + (1 - recent_trending) * rsi_component
        
        # Volatility-targeted position sizing
        # Higher volatility = smaller position to maintain consistent risk
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        raw_signal *= vol_factor
        
        # Apply exponential smoothing
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Adaptive hysteresis to reduce flipping
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            # Higher hysteresis when switching direction
            hysteresis = HYSTERESIS_BASE * 1.5
            if abs(smoothed_signal - prev_signal) < hysteresis:
                smoothed_signal = prev_signal
        
        # Apply minimum signal threshold
        if abs(smoothed_signal) < MIN_SIGNAL_STRENGTH:
            smoothed_signal = 0.0
        
        # Clip to valid range
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals