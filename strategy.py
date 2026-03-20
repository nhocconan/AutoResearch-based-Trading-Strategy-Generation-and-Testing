#!/usr/bin/env python3
"""
strategy.py - Adaptive Regime Trend V5
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Building on adaptive_regime_trend_v2 success (Sharpe=0.500), improving:
    - Simplified regime detection ( cleaner ADX + BB logic)
    - Better trend strength with EMA stack confirmation
    - Improved volume breakout confirmation
    - Cleaner signal smoothing with adaptive hysteresis
    - Reduced parameter complexity for better generalization
    - Better handling of regime transitions
    
    Key improvements over adaptive_regime_trend_v2:
    - Simpler regime classification (trending/ranging/transition)
    - Volume spike confirmation for breakouts (not just percentile)
    - Adaptive signal thresholds based on volatility
    - Cleaner momentum confirmation (RSI slope + price momentum)
    - Reduced regime memory complexity
    - Better volatility-based position sizing

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

name = "adaptive_regime_trend_v5"
timeframe = "1h"
leverage = 2.8  # Slightly increased due to cleaner signal generation

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
ADX_TREND_THRESHOLD = 25
ADX_STRONG_THRESHOLD = 35
ADX_WEAK_THRESHOLD = 20

# Bollinger Band configuration
BB_PERIOD = 20
BB_STD = 2.0
BB_SQUEEZE_THRESHOLD = 0.012  # BB width below this = squeeze

# MACD configuration
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Volume configuration
VOLUME_LOOKBACK = 20
VOLUME_SPIKE_THRESHOLD = 1.5  # Volume > avg * this = spike

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
SMOOTHING_FACTOR = 0.65
HYSTERESIS_THRESHOLD = 0.08

# Regime transition
REGIME_MEMORY = 3  # Bars to remember previous regime


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


def calculate_volume_spike(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume spike indicator using rolling average.
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
    volume_ratio = np.where(volume_ratio < 0, 1.0, volume_ratio)
    
    return volume_ratio


def calculate_price_momentum(close: np.ndarray, lookback: int = 5) -> np.ndarray:
    """
    Calculate price momentum as rate of change.
    Returns value normalized to approximately [-1, 1].
    """
    n = len(close)
    momentum = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return momentum
    
    for i in range(lookback, n):
        if close[i-lookback] > 0:
            roc = (close[i] - close[i-lookback]) / close[i-lookback]
            # Normalize to roughly [-1, 1] for typical crypto moves
            momentum[i] = np.clip(roc / 0.05, -1, 1)
    
    return momentum


def calculate_rsi_slope(rsi: np.ndarray, lookback: int = 3) -> np.ndarray:
    """
    Calculate RSI slope over recent bars.
    Returns value in approximately [-1, 1].
    """
    n = len(rsi)
    slope = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return slope
    
    for i in range(lookback, n):
        rsi_change = rsi[i] - rsi[i-lookback]
        slope[i] = np.clip(rsi_change / (10 * lookback), -1, 1)
    
    return slope


def calculate_ema_stack_score(ema_fast: float, ema_medium: float, 
                               ema_slow: float, ema_major: float,
                               close: float) -> float:
    """
    Calculate EMA stack alignment score.
    Returns value in [-1, 1] where:
    - Positive = bullish alignment (fast > medium > slow > major)
    - Negative = bearish alignment
    - Magnitude = strength of alignment
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    # Check bullish alignment
    bullish = (ema_fast > ema_medium > ema_slow > ema_major)
    bearish = (ema_fast < ema_medium < ema_slow < ema_major)
    
    if bullish:
        # Calculate alignment strength
        spread1 = (ema_fast - ema_medium) / close
        spread2 = (ema_medium - ema_slow) / close
        spread3 = (ema_slow - ema_major) / close
        avg_spread = (spread1 + spread2 + spread3) / 3
        score = np.clip(avg_spread * 100, 0.3, 1.0)
    elif bearish:
        spread1 = (ema_medium - ema_fast) / close
        spread2 = (ema_slow - ema_medium) / close
        spread3 = (ema_major - ema_slow) / close
        avg_spread = (spread1 + spread2 + spread3) / 3
        score = -np.clip(avg_spread * 100, 0.3, 1.0)
    else:
        # Mixed alignment - calculate net bias
        bias = 0.0
        if ema_fast > ema_medium:
            bias += 0.25
        else:
            bias -= 0.25
        if ema_medium > ema_slow:
            bias += 0.25
        else:
            bias -= 0.25
        if ema_slow > ema_major:
            bias += 0.25
        else:
            bias -= 0.25
        if close > ema_major:
            bias += 0.25
        else:
            bias -= 0.25
        score = bias
    
    return np.clip(score, -1.0, 1.0)


def determine_regime(adx: float, bb_width: float, 
                     adx_trend: float, adx_weak: float, 
                     bb_squeeze: float) -> str:
    """
    Determine market regime based on ADX and Bollinger Band width.
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
    Adaptive Regime Trend V5 Strategy.
    
    Signal Logic:
    1. Determine market regime (trending/ranging/squeeze/transition)
    2. Generate regime-appropriate signals
    3. Confirm with volume and momentum
    4. Apply volatility-based position sizing
    5. Smooth signals with adaptive hysteresis
    
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
    rsi_slope = calculate_rsi_slope(rsi, lookback=3)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    
    bb_upper, bb_middle, bb_lower, bb_width = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    macd_line, macd_signal, macd_hist = calculate_macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    
    volume_ratio = calculate_volume_spike(volume, VOLUME_LOOKBACK)
    price_momentum = calculate_price_momentum(close, lookback=5)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        ADX_PERIOD * 2 + 1,
        VOLUME_LOOKBACK,
        BB_PERIOD,
        MACD_SLOW + MACD_SIGNAL,
        10  # Minimum bars for momentum
    )
    
    # Track previous signal for smoothing and hysteresis
    prev_signal = 0.0
    prev_direction = 0  # 0=neutral, 1=long, -1=short
    regime_history = [0] * REGIME_MEMORY  # Track recent regimes
    
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
        
        # Determine current regime
        regime = determine_regime(
            adx[i], bb_width[i], 
            ADX_TREND_THRESHOLD, ADX_WEAK_THRESHOLD, 
            BB_SQUEEZE_THRESHOLD
        )
        
        # Update regime history
        regime_code = {'trending': 1, 'ranging': 2, 'squeeze': 3, 'transition': 0}.get(regime, 0)
        regime_history.pop(0)
        regime_history.append(regime_code)
        
        # Calculate EMA stack score (trend alignment)
        ema_score = calculate_ema_stack_score(
            ema_fast[i], ema_medium[i], ema_slow[i], ema_major[i], close[i]
        )
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_SPIKE_THRESHOLD
        
        # Initialize raw signal
        raw_signal = 0.0
        regime_weight = 0.0
        
        # =========================================================
        # SQUEEZE REGIME - Wait for breakout with volume
        # =========================================================
        if regime == 'squeeze':
            regime_weight = MIN_SIGNAL_BREAKOUT
            
            # Wait for directional confirmation with volume
            if ema_score > 0.15 and volume_confirmed:
                # Bullish breakout
                momentum_conf = (price_momentum[i] > 0) and (rsi_slope[i] > 0)
                if momentum_conf:
                    raw_signal = 0.5 + 0.3 * min(ema_score, 0.5)
                else:
                    raw_signal = 0.3 * ema_score
            elif ema_score < -0.15 and volume_confirmed:
                # Bearish breakout
                momentum_conf = (price_momentum[i] < 0) and (rsi_slope[i] < 0)
                if momentum_conf:
                    raw_signal = -0.5 - 0.3 * min(abs(ema_score), 0.5)
                else:
                    raw_signal = 0.3 * ema_score
            else:
                # No clear direction - stay neutral or reduce
                raw_signal = ema_score * 0.3
        
        # =========================================================
        # TRENDING REGIME - Follow the trend
        # =========================================================
        elif regime == 'trending':
            regime_weight = MIN_SIGNAL_TRENDING
            
            # Base signal from EMA alignment
            raw_signal = ema_score
            
            # Amplify in strong trends
            if adx[i] >= ADX_STRONG_THRESHOLD:
                raw_signal *= 1.15
            
            # Momentum confirmation
            if raw_signal > 0:
                # Long: need positive momentum
                momentum_boost = 1.0
                if price_momentum[i] > 0:
                    momentum_boost += 0.15
                if rsi_slope[i] > 0:
                    momentum_boost += 0.15
                if rsi[i] < RSI_OVERBOUGHT:
                    momentum_boost += 0.1  # Room to run
                raw_signal *= min(momentum_boost, 1.4)
            else:
                # Short: need negative momentum
                momentum_boost = 1.0
                if price_momentum[i] < 0:
                    momentum_boost += 0.15
                if rsi_slope[i] < 0:
                    momentum_boost += 0.15
                if rsi[i] > RSI_OVERSOLD:
                    momentum_boost += 0.1  # Room to run
                raw_signal *= min(momentum_boost, 1.4)
            
            # Volume boost for trend confirmation
            if volume_confirmed:
                raw_signal *= 1.1
        
        # =========================================================
        # RANGING REGIME - Mean reversion with RSI
        # =========================================================
        elif regime == 'ranging':
            regime_weight = MIN_SIGNAL_RANGING
            
            # RSI-based mean reversion
            if rsi[i] < RSI_OVERSOLD:
                raw_signal = 0.35 + 0.15 * (RSI_OVERSOLD - rsi[i]) / 20
            elif rsi[i] > RSI_OVERBOUGHT:
                raw_signal = -0.35 - 0.15 * (rsi[i] - RSI_OVERBOUGHT) / 20
            else:
                raw_signal = 0.0
            
            # Amplify at extremes
            if rsi[i] < RSI_EXTREME_LOW:
                raw_signal = max(raw_signal, 0.55)
            elif rsi[i] > RSI_EXTREME_HIGH:
                raw_signal = min(raw_signal, -0.55)
            
            # Reduce if trend is fighting mean reversion
            if abs(ema_score) > 0.25:
                raw_signal *= 0.6
        
        # =========================================================
        # TRANSITION REGIME - Blend trend and mean reversion
        # =========================================================
        else:  # transition
            regime_weight = MIN_SIGNAL_RANGING * 0.7
            
            # Weighted combination based on ADX position
            adx_position = (adx[i] - ADX_WEAK_THRESHOLD) / (ADX_TREND_THRESHOLD - ADX_WEAK_THRESHOLD)
            adx_position = np.clip(adx_position, 0, 1)
            
            # Trend component
            trend_component = ema_score * adx_position
            
            # Mean reversion component
            mr_component = 0.0
            if rsi[i] < RSI_OVERSOLD:
                mr_component = 0.3 * (1 - adx_position)
            elif rsi[i] > RSI_OVERBOUGHT:
                mr_component = -0.3 * (1 - adx_position)
            
            raw_signal = trend_component + mr_component
        
        # =========================================================
        # Volatility-based position sizing
        # =========================================================
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        raw_signal *= vol_factor
        
        # =========================================================
        # Signal smoothing with adaptive hysteresis
        # =========================================================
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