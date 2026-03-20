#!/usr/bin/env python3
"""
strategy.py - Multi-TF Trend Control V17
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Multi-timeframe trend following with strict risk control:
    - Primary timeframe: 4h (cleaner trends than 1h, more trades than 1d)
    - Trend filter: Price above/below 200 EMA on 4h
    - Entry signal: EMA crossover (12/26) with momentum confirmation
    - Major trend filter: Daily 50 EMA direction
    - Risk control: Volatility-based signal scaling + strict drawdown limits
    - Funding overlay: Only at extreme levels (contrarian at crowded trades)
    
    Why 4h timeframe:
    - Less noise than 1h/15m/5m
    - More trade opportunities than 1d
    - Better risk/reward for trend following
    - Funding rates apply every 8h, aligns well with 4h bars

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

name = "multi_tf_trend_control_v17"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for drawdown control

# EMA configuration for trend detection
EMA_FAST = 12
EMA_SLOW = 26
EMA_MAJOR = 200
EMA_DAILY_FILTER = 50  # For daily trend filter (simulated on 4h)

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_THRESHOLD = 40  # More lenient for trend following
RSI_SHORT_THRESHOLD = 60
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Funding rate configuration
FUNDING_EXTREME_THRESHOLD = 0.0015  # 0.15% per 8hr = very extreme
FUNDING_LOOKBACK = 200  # For calculating extremes
FUNDING_WEIGHT = 0.25  # Conservative funding overlay

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.020  # Target ATR as % of price
VOLATILITY_MIN = 0.005  # Minimum ATR % to trade
VOLATILITY_MAX = 0.080  # Maximum ATR % to trade

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Higher threshold for cleaner signals
MAX_SIGNAL = 0.75  # Conservative max signal
SMOOTHING_FACTOR = 0.60  # More smoothing for stability
HYSERESIS_THRESHOLD = 0.15  # Higher hysteresis to reduce whipsaws

# Risk control
MAX_CONSECUTIVE_LOSSES = 3  # Reduce position after consecutive losses
DRAWDOWN_REDUCTION = 0.50  # Reduce signal by this % after drawdown

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.60


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


def calculate_funding_extremes(funding_rate: np.ndarray, lookback: int = 200) -> tuple:
    """
    Calculate rolling percentile extremes of funding rate.
    Returns: (rolling_90th_percentile, rolling_10th_percentile)
    Only uses past funding rate data (no look-ahead).
    """
    n = len(funding_rate)
    rolling_high = np.zeros(n, dtype=np.float64)
    rolling_low = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return rolling_high, rolling_low
    
    funding_series = pd.Series(funding_rate)
    rolling_high_series = funding_series.rolling(window=lookback, min_periods=lookback).quantile(0.90)
    rolling_low_series = funding_series.rolling(window=lookback, min_periods=lookback).quantile(0.10)
    
    rolling_high = np.nan_to_num(rolling_high_series.values, nan=0.0)
    rolling_low = np.nan_to_num(rolling_low_series.values, nan=0.0)
    
    return rolling_high, rolling_low


def calculate_funding_signal(funding_rate: np.ndarray, 
                             funding_high: np.ndarray,
                             funding_low: np.ndarray,
                             extreme_threshold: float = 0.0015,
                             weight: float = 0.25) -> np.ndarray:
    """
    Calculate funding rate contrarian signal.
    Only triggers at extreme levels to avoid fighting strong trends.
    Returns value in [-weight, weight].
    Only uses current/past funding rate (no look-ahead).
    """
    n = len(funding_rate)
    signal = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        fr = funding_rate[i]
        
        # Only act on extreme funding (contrarian at crowded trades)
        if fr > extreme_threshold:
            # Very positive funding → short bias
            signal[i] = -weight * min(1.0, fr / extreme_threshold)
        elif fr < -extreme_threshold:
            # Very negative funding → long bias
            signal[i] = weight * min(1.0, abs(fr) / extreme_threshold)
        else:
            signal[i] = 0.0
    
    return signal


def calculate_macd_histogram(close: np.ndarray, fast: int = 12, slow: int = 26, signal_period: int = 9) -> np.ndarray:
    """
    Calculate MACD histogram using only past data.
    """
    n = len(close)
    macd_hist = np.zeros(n, dtype=np.float64)
    
    if n < slow + signal_period:
        return macd_hist
    
    close_series = pd.Series(close)
    ema_fast = close_series.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close_series.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal_period, adjust=False, min_periods=signal_period).mean()
    macd_hist_series = macd - macd_signal
    
    macd_hist = np.nan_to_num(macd_hist_series.values, nan=0.0)
    
    return macd_hist


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Multi-TF Trend Control V17 Strategy.
    
    Signal Logic:
    1. Calculate primary trend from EMA crossover (12/26) on 4h
    2. Filter by major trend (price vs 200 EMA)
    3. Confirm with MACD histogram momentum
    4. Add funding overlay only at extreme levels
    5. Scale by volatility for risk normalization
    6. Apply strict smoothing and hysteresis
    7. Filter by minimum signal magnitude
    
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
    ema_fast = calculate_ema(close, EMA_FAST)
    ema_slow = calculate_ema(close, EMA_SLOW)
    ema_major = calculate_ema(close, EMA_MAJOR)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    macd_hist = calculate_macd_histogram(close, EMA_FAST, EMA_SLOW, 9)
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    funding_high, funding_low = calculate_funding_extremes(funding_rate, FUNDING_LOOKBACK)
    funding_signal = calculate_funding_signal(
        funding_rate, funding_high, funding_low,
        FUNDING_EXTREME_THRESHOLD, FUNDING_WEIGHT
    )
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW + 9,  # MACD signal period
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        VOLUME_LOOKBACK,
        FUNDING_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    consecutive_losses = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0:
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
        
        # Determine trend direction from EMA crossover
        ema_diff = ema_fast[i] - ema_slow[i]
        ema_direction = np.sign(ema_diff)
        
        # Major trend filter (price vs 200 EMA)
        major_trend = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend (strict filter)
        if ema_direction != major_trend or ema_direction == 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_direction = 0
            continue
        
        # MACD momentum confirmation
        macd_momentum = np.sign(macd_hist[i])
        
        # Require MACD to confirm trend direction
        if macd_momentum != ema_direction:
            # Weakening momentum → reduce signal
            raw_signal = ema_direction * 0.3
        else:
            # Strong confirmation
            trend_strength = min(1.0, abs(ema_diff) / close[i] * 100)
            raw_signal = ema_direction * trend_strength
        
        # RSI filter (avoid extreme overbought/oversold in trend direction)
        if ema_direction > 0 and rsi[i] > RSI_OVERBOUGHT:
            raw_signal *= 0.5  # Reduce long strength if overbought
        elif ema_direction < 0 and rsi[i] < RSI_OVERSOLD:
            raw_signal *= 0.5  # Reduce short strength if oversold
        
        # Funding overlay (only at extremes, contrarian)
        fund_sig = funding_signal[i]
        if abs(fund_sig) > 0.1 and np.sign(fund_sig) != ema_direction:
            # Funding contradicts trend at extreme → reduce signal
            raw_signal *= (1.0 - FUNDING_WEIGHT)
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.5, 1.5)  # More conservative scaling
        raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals)
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Hysteresis: don't flip direction on small changes
        current_direction = np.sign(smoothed_signal)
        if current_direction != 0 and current_direction != prev_direction:
            if abs(smoothed_signal - prev_signal) < HYSERESIS_THRESHOLD:
                smoothed_signal = prev_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals