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
    Building on adaptive_regime_trend_v3 success (Sharpe=0.531), improving:
    - Taker buy/sell ratio for market pressure detection
    - Cleaner regime detection with smoother transitions
    - Better funding rate mean reversion integration
    - Improved volatility-adaptive position sizing
    - Reduced signal flipping with better hysteresis
    - More conservative in uncertain regimes
    
    Key improvements over adaptive_regime_trend_v3:
    - Taker ratio integration: taker_buy_volume/volume as pressure signal
    - Simplified regime detection (trending vs ranging with confidence)
    - Funding rate as mean reversion overlay (not primary signal)
    - Better volatility scaling (inverse ATR relationship)
    - Stronger hysteresis to reduce whipsaw
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

name = "adaptive_regime_trend_v5"
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
ADX_STRONG_THRESHOLD = 40

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
VOLATILITY_TARGET = 0.008
VOLATILITY_MIN = 0.002
VOLATILITY_MAX = 0.035

# Signal configuration
MIN_SIGNAL_TRENDING = 0.20
MIN_SIGNAL_RANGING = 0.30
MAX_SIGNAL = 0.70
SMOOTHING_FACTOR = 0.70
HYSTERESIS_THRESHOLD = 0.10

# Funding rate configuration
FUNDING_EXTREME_THRESHOLD = 0.0005  # 0.05% per 8h
FUNDING_BIAS_WEIGHT = 0.20

# Taker ratio configuration
TAKER_RATIO_LOOKBACK = 20
TAKER_RATIO_THRESHOLD = 0.55  # Above this = bullish pressure


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


def calculate_taker_ratio(prices: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """
    Calculate taker buy/sell ratio from volume data.
    Returns rolling average of taker buy pressure.
    Only uses past data (no look-ahead).
    
    Looks for taker_buy_volume column, falls back to 0.5 if not available.
    """
    n = len(prices)
    taker_ratio = np.full(n, 0.5, dtype=np.float64)
    
    try:
        # Try to get taker buy volume
        taker_buy = prices["taker_buy_volume"].values.astype(np.float64)
        volume = prices["volume"].values.astype(np.float64)
        
        taker_buy = np.nan_to_num(taker_buy, nan=0.0)
        volume = np.nan_to_num(volume, nan=1.0)
        volume = np.where(volume <= 0, 1.0, volume)
        
        # Calculate instantaneous taker ratio
        inst_ratio = taker_buy / volume
        inst_ratio = np.clip(inst_ratio, 0.0, 1.0)
        
        # Rolling average
        inst_series = pd.Series(inst_ratio)
        rolling_ratio = inst_series.rolling(window=lookback, min_periods=lookback).mean()
        
        taker_ratio = np.nan_to_num(rolling_ratio.values, nan=0.5)
        
    except (KeyError, TypeError, ValueError):
        # Column not available, use neutral value
        pass
    
    return taker_ratio


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


def calculate_trend_direction(close: float, ema_fast: float, ema_medium: float, 
                              ema_slow: float, ema_major: float) -> float:
    """
    Calculate trend direction score based on EMA alignment.
    Returns value in [-1, 1] where sign indicates direction.
    """
    if close <= 0 or ema_major <= 0:
        return 0.0
    
    # Count bullish alignments
    bullish_count = 0
    bearish_count = 0
    
    if ema_fast > ema_medium:
        bullish_count += 1
    else:
        bearish_count += 1
        
    if ema_medium > ema_slow:
        bullish_count += 1
    else:
        bearish_count += 1
        
    if ema_slow > ema_major:
        bullish_count += 1
    else:
        bearish_count += 1
    
    if close > ema_major:
        bullish_count += 1
    else:
        bearish_count += 1
    
    # Net direction
    net = bullish_count - bearish_count
    direction = net / 4.0  # Normalize to [-1, 1]
    
    return direction


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Adaptive Regime Trend V5 Strategy.
    
    Signal Logic:
    1. Calculate regime confidence (trending vs ranging)
    2. Trending: Follow EMA stack direction with MACD confirmation
    3. Ranging: RSI mean reversion with Bollinger Band support
    4. Funding rate as mean reversion overlay
    5. Taker ratio for market pressure confirmation
    6. Volatility-adaptive position sizing
    7. Signal smoothing with hysteresis
    
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
    
    volume_ratio = calculate_volume_spike(volume, VOLUME_LOOKBACK)
    taker_ratio = calculate_taker_ratio(prices, TAKER_RATIO_LOOKBACK)
    funding_bias = calculate_funding_bias(funding_rate, FUNDING_EXTREME_THRESHOLD)
    
    # Determine minimum valid index
    min_valid_index = max(
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        ADX_PERIOD * 2 + 1,
        VOLUME_LOOKBACK,
        BB_PERIOD,
        MACD_SLOW + MACD_SIGNAL,
        TAKER_RATIO_LOOKBACK
    )
    
    # Track previous signal for smoothing and hysteresis
    prev_signal = 0.0
    prev_direction = 0  # 0=neutral, 1=long, -1=short
    
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
        is_trending = adx[i] >= ADX_TREND_THRESHOLD
        is_squeeze = bb_width[i] < BB_SQUEEZE_THRESHOLD
        
        # Calculate trend direction
        trend_dir = calculate_trend_direction(
            close[i], ema_fast[i], ema_medium[i],
            ema_slow[i], ema_major[i]
        )
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_SPIKE_THRESHOLD
        
        # Taker pressure signal
        taker_signal = 0.0
        if taker_ratio[i] > TAKER_RATIO_THRESHOLD:
            taker_signal = 0.3  # Bullish pressure
        elif taker_ratio[i] < (1.0 - TAKER_RATIO_THRESHOLD):
            taker_signal = -0.3  # Bearish pressure
        
        # Initialize raw signal
        raw_signal = 0.0
        regime_weight = 0.0
        
        # TRENDING REGIME
        if is_trending and not is_squeeze:
            regime_weight = MIN_SIGNAL_TRENDING
            
            # Base signal from trend direction
            raw_signal = trend_dir
            
            # Amplify in strong trends
            if adx[i] >= ADX_STRONG_THRESHOLD:
                raw_signal *= 1.15
            
            # MACD confirmation
            macd_conf = np.sign(macd_hist[i])
            if np.sign(trend_dir) == macd_conf:
                raw_signal *= 1.1  # Confirming
            else:
                raw_signal *= 0.85  # Diverging
            
            # Volume boost
            if volume_confirmed:
                raw_signal *= 1.1
            
            # Taker ratio confirmation
            if np.sign(trend_dir) == np.sign(taker_signal):
                raw_signal += taker_signal * 0.3
            
            # Apply funding bias (reduce position if funding opposes trend)
            if abs(funding_bias[i]) > 0.3:
                if np.sign(trend_dir) != np.sign(funding_bias[i]):
                    raw_signal *= (1.0 - FUNDING_BIAS_WEIGHT)
        
        # RANGING REGIME (mean reversion)
        elif not is_trending:
            regime_weight = MIN_SIGNAL_RANGING
            
            # RSI-based mean reversion
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
            
            # Bollinger Band confirmation
            if raw_signal > 0 and close[i] < bb_lower[i]:
                raw_signal *= 1.15  # Price below lower band, stronger long
            elif raw_signal < 0 and close[i] > bb_upper[i]:
                raw_signal *= 1.15  # Price above upper band, stronger short
            
            # Apply funding bias (enhance mean reversion)
            if abs(funding_bias[i]) > 0.3:
                if np.sign(raw_signal) == np.sign(funding_bias[i]):
                    raw_signal *= (1.0 + FUNDING_BIAS_WEIGHT)
            
            # Taker ratio for entry timing
            if raw_signal > 0 and taker_signal > 0:
                raw_signal += taker_signal * 0.2
            elif raw_signal < 0 and taker_signal < 0:
                raw_signal += taker_signal * 0.2
        
        # SQUEEZE REGIME (wait for breakout)
        else:
            # Squeeze with trending ADX = potential breakout
            regime_weight = MIN_SIGNAL_TRENDING * 0.5
            
            # Wait for directional confirmation
            if trend_dir > 0.3 and volume_confirmed:
                raw_signal = trend_dir * 0.6
            elif trend_dir < -0.3 and volume_confirmed:
                raw_signal = trend_dir * 0.6
            else:
                raw_signal = 0.0
        
        # Volatility-based position sizing (inverse relationship)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 1.8)
        
        raw_signal *= vol_factor
        
        # Apply exponential smoothing
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Apply hysteresis to reduce flipping
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
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