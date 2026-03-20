#!/usr/bin/env python3
"""
strategy.py - Mean Reversion Bollinger V1
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "15m")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Intraday mean reversion on 15m timeframe:
    - Core signal: Bollinger Band mean reversion (price vs 20 SMA ± 2 std)
    - Confirmation: RSI divergence (oversold/overbought conditions)
    - Trend filter: 200 EMA to avoid fighting major trends
    - Volatility adjustment: ATR-based position sizing
    - Risk controls: Signal capping, hysteresis, volatility normalization
    
    Why 15m timeframe:
    - More trades than 1h/4h (better statistical significance)
    - Less noise than 1m/5m (cleaner signals)
    - Captures intraday mean reversion patterns well
    - Funding rate impact still relevant
    
    Why mean reversion:
    - Crypto exhibits strong mean-reverting behavior intraday
    - After large moves, price tends to revert to mean
    - Lower drawdown than pure trend following
    - Better Sharpe ratio potential with proper risk controls

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

name = "mean_reversion_bollinger_v1"
timeframe = "15m"
leverage = 1.5  # Conservative leverage for drawdown control

# Bollinger Band configuration
BB_PERIOD = 20
BB_STD_MULTIPLIER = 2.0

# RSI configuration for confirmation
RSI_PERIOD = 14
RSI_OVERSOLD = 35  # Below this → long signal
RSI_OVERBOUGHT = 65  # Above this → short signal
RSI_NEUTRAL_LOW = 40
RSI_NEUTRAL_HIGH = 60

# EMA trend filter
EMA_MAJOR = 200
TREND_FILTER_STRENGTH = 0.5  # How much trend filter affects signal

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012  # Target ATR as % of price
VOLATILITY_MIN = 0.002  # Minimum ATR % to trade
VOLATILITY_MAX = 0.040  # Maximum ATR % to trade

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.12  # Minimum signal to generate position
MAX_SIGNAL = 0.60  # Maximum signal magnitude (conservative)
SMOOTHING_FACTOR = 0.40  # EMA smoothing for signals
HYSTERESIS_THRESHOLD = 0.08  # Minimum change to flip signal direction

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.60  # Volume must be at least this % of average

# Funding rate configuration (contrarian overlay)
FUNDING_EXTREME_THRESHOLD = 0.0008  # 0.08% per 8hr
FUNDING_LOOKBACK = 80
FUNDING_WEIGHT = 0.25  # How much funding affects signal


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_sma(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average using only past data.
    """
    n = len(close)
    sma = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return sma
    
    close_series = pd.Series(close)
    sma_values = close_series.rolling(window=period, min_periods=period).mean().values
    sma = np.nan_to_num(sma_values, nan=0.0)
    
    return sma


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


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_multiplier: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands using only past data.
    Returns: (middle_band, upper_band, lower_band)
    """
    n = len(close)
    middle = np.zeros(n, dtype=np.float64)
    upper = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return middle, upper, lower
    
    close_series = pd.Series(close)
    middle_series = close_series.rolling(window=period, min_periods=period).mean()
    std_series = close_series.rolling(window=period, min_periods=period).std()
    
    middle = np.nan_to_num(middle_series.values, nan=0.0)
    upper = np.nan_to_num((middle_series + std_multiplier * std_series).values, nan=0.0)
    lower = np.nan_to_num((middle_series - std_multiplier * std_series).values, nan=0.0)
    
    return middle, upper, lower


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


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio vs rolling average.
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


def calculate_funding_signal(funding_rate: np.ndarray, 
                             lookback: int = 80,
                             extreme_threshold: float = 0.0008,
                             weight: float = 0.25) -> np.ndarray:
    """
    Calculate funding rate contrarian signal.
    Extreme positive funding → short bias (negative signal)
    Extreme negative funding → long bias (positive signal)
    Returns value in [-weight, weight].
    Only uses current/past funding rate (no look-ahead).
    """
    n = len(funding_rate)
    signal = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return signal
    
    funding_series = pd.Series(funding_rate)
    rolling_high = funding_series.rolling(window=lookback, min_periods=lookback).quantile(0.90)
    rolling_low = funding_series.rolling(window=lookback, min_periods=lookback).quantile(0.10)
    
    for i in range(lookback, n):
        fr = funding_rate[i]
        
        # Contrarian signal: extreme funding → opposite position
        if fr > extreme_threshold:
            signal[i] = -weight * min(1.0, fr / extreme_threshold)
        elif fr < -extreme_threshold:
            signal[i] = weight * min(1.0, abs(fr) / extreme_threshold)
        elif rolling_high.iloc[i] > 0 and fr > rolling_high.iloc[i] * 0.85:
            signal[i] = -weight * 0.5
        elif rolling_low.iloc[i] < 0 and fr < rolling_low.iloc[i] * 0.85:
            signal[i] = weight * 0.5
    
    return signal


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Mean Reversion Bollinger V1 Strategy.
    
    Signal Logic:
    1. Calculate Bollinger Bands (20 SMA ± 2 std)
    2. Calculate RSI for overbought/oversold confirmation
    3. Calculate 200 EMA for trend filter
    4. Generate mean reversion signal:
       - Price below lower BB + RSI oversold → long
       - Price above upper BB + RSI overbought → short
    5. Apply trend filter (reduce signal against major trend)
    6. Add funding rate contrarian overlay
    7. Apply volatility normalization
    8. Smooth signals with EMA
    9. Apply hysteresis to reduce whipsaws
    10. Filter by minimum signal magnitude
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, funding_rate, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract price data with error handling
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
    
    # Clean data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Fix invalid prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators (all use only past data)
    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD_MULTIPLIER)
    ema_major = calculate_ema(close, EMA_MAJOR)
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    funding_signal = calculate_funding_signal(
        funding_rate, FUNDING_LOOKBACK, FUNDING_EXTREME_THRESHOLD, FUNDING_WEIGHT
    )
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        BB_PERIOD,
        EMA_MAJOR,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        FUNDING_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0 or bb_middle[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volatility filter (not too low, not too high)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Volume filter (ensure sufficient liquidity)
        if volume_ratio[i] < VOLUME_MIN_RATIO:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Calculate Bollinger Band position (% between lower and upper)
        bb_range = bb_upper[i] - bb_lower[i]
        if bb_range <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        bb_position = (close[i] - bb_lower[i]) / bb_range  # 0=lower, 1=upper, 0.5=middle
        
        # Calculate mean reversion signal from Bollinger Bands
        # Price near lower band → long signal, near upper band → short signal
        bb_signal = 0.0
        if bb_position < 0.15:  # Price in bottom 15% of BB range
            bb_signal = (0.15 - bb_position) / 0.15  # 0 to 1
        elif bb_position > 0.85:  # Price in top 15% of BB range
            bb_signal = -(bb_position - 0.85) / 0.15  # -1 to 0
        
        # RSI confirmation
        rsi_signal = 0.0
        if rsi[i] < RSI_OVERSOLD:
            rsi_signal = (RSI_OVERSOLD - rsi[i]) / RSI_OVERSOLD  # Positive for oversold
        elif rsi[i] > RSI_OVERBOUGHT:
            rsi_signal = -(rsi[i] - RSI_OVERBOUGHT) / (100 - RSI_OVERBOUGHT)  # Negative for overbought
        
        # Combine BB and RSI signals (both must agree for strong signal)
        if np.sign(bb_signal) == np.sign(rsi_signal) or abs(bb_signal) > 0.5 or abs(rsi_signal) > 0.5:
            raw_signal = bb_signal * 0.6 + rsi_signal * 0.4
        else:
            # Conflicting signals → reduce strength
            raw_signal = bb_signal * 0.4 + rsi_signal * 0.2
        
        # Trend filter: reduce signal against major trend
        if ema_major[i] > 0:
            trend_direction = np.sign(close[i] - ema_major[i])
            if np.sign(raw_signal) != trend_direction and abs(raw_signal) > 0.3:
                raw_signal *= (1.0 - TREND_FILTER_STRENGTH)
        
        # Add funding rate contrarian overlay
        raw_signal += funding_signal[i]
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 1.8)
        raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Hysteresis: don't flip direction on small changes
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < HYSTERESIS_THRESHOLD:
                smoothed_signal = prev_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal (conservative)
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals