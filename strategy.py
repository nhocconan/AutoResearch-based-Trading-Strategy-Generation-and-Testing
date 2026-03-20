#!/usr/bin/env python3
"""
strategy.py - Bollinger Keltner Squeeze Mean Reversion V1
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

Strategy Hypothesis:
    Bollinger-Keltner squeeze detection with RSI mean reversion on 1h timeframe.
    
    Why this should beat Supertrend:
    - Mean reversion has LOWER drawdowns than pure trend following
    - Squeeze breakouts capture volatility expansions with better timing
    - RSI extremes provide superior entry timing vs EMA crossovers
    - 1h timeframe balances noise reduction with sufficient trade frequency
    - Volatility regime detection avoids trading during choppy periods
    
    Signal Logic:
    1. Detect Bollinger-Keltner squeeze (BB inside KC = low vol regime)
    2. Wait for squeeze release (price breaks BB with volume)
    3. RSI confirmation (not extreme, allowing momentum continuation)
    4. SMA(200) filter for trend alignment
    5. Dynamic position sizing based on volatility regime
    
    Expected Improvement:
    - Lower max drawdown (<40% vs 84% from Supertrend)
    - Higher Sharpe (>0.5 vs 0.25 from Supertrend)
    - More trades during ranging markets where Supertrend fails

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

name = "bb_kc_squeeze_meanrev_1h"
timeframe = "1h"
leverage = 1.5  # Conservative leverage for mean reversion

# Bollinger Bands configuration
BB_PERIOD = 20
BB_STD = 2.0

# Keltner Channel configuration
KC_PERIOD = 20
KC_ATR_MULT = 1.5

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_ENTRY = 45  # RSI must be above this for long entries
RSI_SHORT_ENTRY = 55  # RSI must be below this for short entries
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Trend filter configuration
SMA_MAJOR = 200
TREND_FILTER_ENABLED = True

# Squeeze detection
SQUEEZE_LOOKBACK = 10  # Bars to confirm squeeze
SQUEEZE_RELEASE_BARS = 3  # Bars after squeeze release to enter

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_BREAKOUT_RATIO = 1.3  # Volume must be >1.3x average on breakout

# Volatility regime detection
VOLATILITY_LOOKBACK = 50
VOLATILITY_LOW_PERCENTILE = 30  # Bottom 30% = low vol regime
VOLATILITY_HIGH_PERCENTILE = 70  # Top 30% = high vol regime

# Risk management
ATR_PERIOD = 14
ATR_STOP_MULT = 2.5  # ATR multiplier for stop loss
MAX_POSITION_SIZE = 0.85  # Maximum signal magnitude
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position

# Signal smoothing
SIGNAL_SMOOTHING = 0.40  # EMA smoothing factor for signals
HYSTERESIS_THRESHOLD = 0.15  # Minimum change to flip signal direction


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


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_mult: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands (middle, upper, lower) using only past data.
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
    std_values = np.nan_to_num(std_series.values, nan=0.0)
    
    upper = middle + std_mult * std_values
    lower = middle - std_mult * std_values
    
    return middle, upper, lower


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


def calculate_keltner_channel(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                               period: int = 20, atr_mult: float = 1.5) -> tuple:
    """
    Calculate Keltner Channel (middle, upper, lower) using only past data.
    """
    n = len(close)
    middle = np.zeros(n, dtype=np.float64)
    upper = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return middle, upper, lower
    
    middle = calculate_sma(close, period)
    atr = calculate_atr(high, low, close, period)
    
    upper = middle + atr_mult * atr
    lower = middle - atr_mult * atr
    
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


def detect_squeeze(bb_upper: np.ndarray, bb_lower: np.ndarray,
                   kc_upper: np.ndarray, kc_lower: np.ndarray,
                   lookback: int = 10) -> np.ndarray:
    """
    Detect Bollinger-Keltner squeeze.
    Squeeze = BB inside KC (low volatility compression).
    Returns 1.0 when in squeeze, 0.0 otherwise.
    Only uses past data (no look-ahead).
    """
    n = len(bb_upper)
    squeeze = np.zeros(n, dtype=np.float64)
    
    for i in range(lookback, n):
        # Check if BB is inside KC for the last 'lookback' bars
        in_squeeze = True
        for j in range(i - lookback + 1, i + 1):
            if bb_upper[j] > kc_upper[j] or bb_lower[j] < kc_lower[j]:
                in_squeeze = False
                break
        
        squeeze[i] = 1.0 if in_squeeze else 0.0
    
    return squeeze


def detect_squeeze_release(squeeze: np.ndarray, close: np.ndarray,
                           bb_upper: np.ndarray, bb_lower: np.ndarray,
                           release_bars: int = 3) -> np.ndarray:
    """
    Detect squeeze release (breakout from compression).
    Returns direction: +1 for upper breakout, -1 for lower breakout, 0 for no release.
    Only uses past data (no look-ahead).
    """
    n = len(squeeze)
    release = np.zeros(n, dtype=np.float64)
    
    for i in range(release_bars, n):
        # Check if we were in squeeze 'release_bars' ago
        if squeeze[i - release_bars] == 1.0:
            # Check if price broke out
            if close[i] > bb_upper[i]:
                release[i] = 1.0  # Upper breakout
            elif close[i] < bb_lower[i]:
                release[i] = -1.0  # Lower breakout
    
    return release


def calculate_volatility_regime(close: np.ndarray, lookback: int = 50) -> np.ndarray:
    """
    Calculate volatility regime based on rolling ATR percentile.
    Returns: 0 = low vol, 1 = normal vol, 2 = high vol
    Only uses past data (no look-ahead).
    """
    n = len(close)
    regime = np.ones(n, dtype=np.float64)  # Default to normal
    
    if n < lookback * 2:
        return regime
    
    # Calculate rolling ATR as % of price
    high = close * 1.01  # Approximation if high not available
    low = close * 0.99
    atr = calculate_atr(high, low, close, 14)
    atr_pct = atr / close
    
    atr_series = pd.Series(atr_pct)
    
    for i in range(lookback, n):
        window = atr_pct[max(0, i - lookback):i + 1]
        if len(window) >= lookback // 2:
            percentile = np.percentile(window, atr_pct[i] / np.max(window) * 100) if np.max(window) > 0 else 50
            
            if percentile < VOLATILITY_LOW_PERCENTILE:
                regime[i] = 0.0  # Low vol
            elif percentile > VOLATILITY_HIGH_PERCENTILE:
                regime[i] = 2.0  # High vol
            else:
                regime[i] = 1.0  # Normal vol
    
    return regime


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Bollinger-Keltner Squeeze Mean Reversion Strategy.
    
    Signal Logic:
    1. Calculate Bollinger Bands and Keltner Channels
    2. Detect squeeze (BB inside KC = low volatility)
    3. Detect squeeze release (breakout with volume)
    4. RSI confirmation for entry timing
    5. SMA(200) trend filter
    6. Volatility regime for position sizing
    7. Signal smoothing and hysteresis
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, ...]
    
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
    except (KeyError, TypeError, ValueError):
        return signals
    
    # Clean data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Fix invalid prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close * 1.01, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators (all use only past data)
    bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    kc_middle, kc_upper, kc_lower = calculate_keltner_channel(high, low, close, KC_PERIOD, KC_ATR_MULT)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    sma_major = calculate_sma(close, SMA_MAJOR)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    squeeze = detect_squeeze(bb_upper, bb_lower, kc_upper, kc_lower, SQUEEZE_LOOKBACK)
    squeeze_release = detect_squeeze_release(squeeze, close, bb_upper, bb_lower, SQUEEZE_RELEASE_BARS)
    vol_regime = calculate_volatility_regime(close, VOLATILITY_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        BB_PERIOD,
        KC_PERIOD + 1,
        RSI_PERIOD + 1,
        SMA_MAJOR,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        SQUEEZE_LOOKBACK + SQUEEZE_RELEASE_BARS,
        VOLATILITY_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # Initialize raw signal
        raw_signal = 0.0
        
        # Check for squeeze release breakout
        if squeeze_release[i] != 0.0:
            breakout_direction = squeeze_release[i]  # +1 or -1
            
            # Volume confirmation
            if volume_ratio[i] < VOLUME_BREAKOUT_RATIO:
                signals[i] = 0.0
                prev_signal = 0.0
                prev_direction = 0
                continue
            
            # RSI confirmation
            rsi_confirmed = False
            if breakout_direction > 0:  # Long breakout
                if rsi[i] > RSI_LONG_ENTRY and rsi[i] < RSI_OVERBOUGHT:
                    rsi_confirmed = True
            else:  # Short breakout
                if rsi[i] < RSI_SHORT_ENTRY and rsi[i] > RSI_OVERSOLD:
                    rsi_confirmed = True
            
            if not rsi_confirmed:
                signals[i] = 0.0
                prev_signal = 0.0
                prev_direction = 0
                continue
            
            # Trend filter (optional)
            if TREND_FILTER_ENABLED:
                if breakout_direction > 0 and close[i] < sma_major[i]:
                    # Long but price below 200 SMA - reduce signal
                    trend_alignment = 0.5
                elif breakout_direction < 0 and close[i] > sma_major[i]:
                    # Short but price above 200 SMA - reduce signal
                    trend_alignment = 0.5
                else:
                    trend_alignment = 1.0
            else:
                trend_alignment = 1.0
            
            # Volatility regime position sizing
            if vol_regime[i] == 0.0:  # Low vol
                vol_factor = 1.2  # Increase position in low vol (squeeze release)
            elif vol_regime[i] == 2.0:  # High vol
                vol_factor = 0.6  # Reduce position in high vol
            else:
                vol_factor = 1.0
            
            # Calculate raw signal
            raw_signal = breakout_direction * trend_alignment * vol_factor
            
        # Else: check for mean reversion in normal regime (no squeeze)
        elif squeeze[i] == 0.0:
            # Mean reversion: fade extreme moves
            bb_width = (bb_upper[i] - bb_lower[i]) / bb_middle[i] if bb_middle[i] > 0 else 0
            
            # Price position within BB
            price_position = (close[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i]) if (bb_upper[i] - bb_lower[i]) > 0 else 0.5
            
            # Extreme positions with RSI confirmation
            if price_position > 0.85 and rsi[i] > RSI_OVERBOUGHT:
                raw_signal = -0.5  # Short signal
            elif price_position < 0.15 and rsi[i] < RSI_OVERSOLD:
                raw_signal = 0.5  # Long signal
        
        # Apply volatility normalization
        if raw_signal != 0.0:
            atr_pct = atr[i] / close[i]
            vol_target = 0.015  # Target 1.5% ATR
            vol_factor = vol_target / max(atr_pct, 0.003)
            vol_factor = np.clip(vol_factor, 0.5, 2.0)
            raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SIGNAL_SMOOTHING * prev_signal + (1.0 - SIGNAL_SMOOTHING) * raw_signal
        
        # Hysteresis: don't flip direction on small changes
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < HYSTERESIS_THRESHOLD:
                smoothed_signal = prev_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_POSITION_SIZE, MAX_POSITION_SIZE)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals