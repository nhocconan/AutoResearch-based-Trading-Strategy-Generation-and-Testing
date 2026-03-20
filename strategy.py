#!/usr/bin/env python3
"""
strategy.py - Volatility Breakout with Momentum Confirmation
=======================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Volatility breakout strategy with momentum and volume confirmation.
    - Detect Bollinger Band squeeze (low volatility periods)
    - Enter on breakout with volume confirmation
    - Filter by longer-term trend direction (200 EMA)
    - Use ADX to avoid choppy markets
    - ATR-based position sizing for volatility adjustment

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

name = "volatility_breakout_momentum"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for breakout strategy

# Strategy parameters
BB_PERIOD = 20              # Bollinger Bands period
BB_STD = 2.0                # Bollinger Bands standard deviations
SQUEEZE_THRESHOLD = 0.5     # BB width percentile for squeeze detection
MOMENTUM_PERIOD = 10        # Rate of Change period
VOLUME_LOOKBACK = 20        # Lookback for volume average
VOLUME_THRESHOLD = 1.3      # Volume spike multiplier
TREND_EMA = 200             # Long-term trend filter EMA
ADX_PERIOD = 14             # ADX calculation period
ADX_MIN = 20                # Minimum ADX for trending market
ATR_PERIOD = 14             # ATR calculation period
MIN_SIGNAL = 0.25           # Minimum signal magnitude to trade


# =============================================================================
# Technical Indicator Calculations
# =============================================================================

def calculate_sma(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Simple Moving Average using only past data.
    
    Args:
        close: Array of close prices
        period: SMA period
    
    Returns:
        Array of SMA values
    """
    n = len(close)
    sma = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return sma
    
    # Calculate SMA using rolling window
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i - period + 1:i + 1])
    
    return sma


def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
    
    Args:
        close: Array of close prices
        period: EMA period
    
    Returns:
        Array of EMA values
    """
    n = len(close)
    ema = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return ema
    
    # Initialize with SMA
    ema[period - 1] = np.mean(close[:period])
    
    # Calculate EMA multiplier
    multiplier = 2.0 / (period + 1)
    
    # Calculate EMA for remaining periods
    for i in range(period, n):
        ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple:
    """
    Calculate Bollinger Bands using only past data.
    
    Args:
        close: Array of close prices
        period: BB period
        std_dev: Standard deviation multiplier
    
    Returns:
        Tuple of (upper, middle, lower, bandwidth) arrays
    """
    n = len(close)
    upper = np.zeros(n, dtype=np.float64)
    middle = np.zeros(n, dtype=np.float64)
    lower = np.zeros(n, dtype=np.float64)
    bandwidth = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return upper, middle, lower, bandwidth
    
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        middle[i] = np.mean(window)
        std = np.std(window, ddof=0)
        upper[i] = middle[i] + std_dev * std
        lower[i] = middle[i] - std_dev * std
        bandwidth[i] = (upper[i] - lower[i]) / middle[i] if middle[i] > 0 else 0
    
    return upper, middle, lower, bandwidth


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using only past data.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ATR period
    
    Returns:
        Array of ATR values
    """
    n = len(close)
    atr = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return atr
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Initialize ATR with SMA of TR
    atr[period - 1] = np.mean(tr[:period])
    
    # Calculate ATR using Wilder's smoothing
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index using only past data.
    
    Args:
        close: Array of close prices
        period: RSI period
    
    Returns:
        Array of RSI values (0-100)
    """
    n = len(close)
    rsi = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    delta = np.diff(close)
    
    # Separate gains and losses
    gains = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        if delta[i-1] > 0:
            gains[i] = delta[i-1]
        else:
            losses[i] = -delta[i-1]
    
    # Calculate average gains and losses using EMA
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    
    # Initialize with SMA
    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])
    
    # Calculate using Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    
    # Calculate RSI
    for i in range(period, n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average Directional Index using only past data.
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        period: ADX period
    
    Returns:
        Array of ADX values (0-100)
    """
    n = len(close)
    adx = np.zeros(n, dtype=np.float64)
    
    if n < period * 2:
        return adx
    
    # Calculate True Range
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Calculate Directional Movement
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    
    for i in range(1, n):
        plus_move = high[i] - high[i-1]
        minus_move = low[i-1] - low[i]
        
        if plus_move > minus_move and plus_move > 0:
            plus_dm[i] = plus_move
        if minus_move > plus_move and minus_move > 0:
            minus_dm[i] = minus_move
    
    # Smooth TR, +DM, -DM using Wilder's method
    atr = np.zeros(n, dtype=np.float64)
    plus_di = np.zeros(n, dtype=np.float64)
    minus_di = np.zeros(n, dtype=np.float64)
    
    # Initialize with SMA
    atr[period-1] = np.mean(tr[:period])
    plus_di[period-1] = 100 * np.mean(plus_dm[:period]) / atr[period-1] if atr[period-1] > 0 else 0
    minus_di[period-1] = 100 * np.mean(minus_dm[:period]) / atr[period-1] if atr[period-1] > 0 else 0
    
    # Smooth using Wilder's method
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        plus_di[i] = 100 * ((plus_di[i-1] * (period - 1) / 100) * (period - 1) + plus_dm[i]) / period / atr[i] if atr[i] > 0 else 0
        minus_di[i] = 100 * ((minus_di[i-1] * (period - 1) / 100) * (period - 1) + minus_dm[i]) / period / atr[i] if atr[i] > 0 else 0
    
    # Calculate DX and ADX
    dx = np.zeros(n, dtype=np.float64)
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Smooth DX to get ADX
    adx[period*2-1] = np.mean(dx[period:period*2])
    for i in range(period*2, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
    
    return adx


def calculate_momentum(close: np.ndarray, period: int = 10) -> np.ndarray:
    """
    Calculate Rate of Change (momentum) using only past data.
    
    Args:
        close: Array of close prices
        period: Momentum period
    
    Returns:
        Array of momentum values (percentage)
    """
    n = len(close)
    momentum = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return momentum
    
    for i in range(period, n):
        if close[i - period] > 0:
            momentum[i] = (close[i] - close[i - period]) / close[i - period] * 100
    
    return momentum


def calculate_volume_ratio(volume: np.ndarray, lookback: int = 20) -> np.ndarray:
    """
    Calculate volume ratio relative to rolling average.
    Only uses past volume data (no look-ahead).
    
    Args:
        volume: Array of volume values
        lookback: Rolling window for average calculation
    
    Returns:
        Array of volume ratios
    """
    n = len(volume)
    volume_ratio = np.ones(n, dtype=np.float64)
    
    if n < lookback:
        return volume_ratio
    
    for i in range(lookback - 1, n):
        avg_volume = np.mean(volume[i - lookback + 1:i + 1])
        if avg_volume > 0:
            volume_ratio[i] = volume[i] / avg_volume
    
    return volume_ratio


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Volatility Breakout Strategy with Momentum Confirmation.
    
    Signal Logic:
    1. Detect Bollinger Band squeeze (low volatility)
    2. Wait for breakout above/below bands
    3. Confirm with volume spike and momentum
    4. Filter by long-term trend (200 EMA)
    5. Filter by ADX (avoid choppy markets)
    
    Entry Conditions:
    - LONG: Price breaks above BB upper + volume spike + momentum positive + trend up
    - SHORT: Price breaks below BB lower + volume spike + momentum negative + trend down
    
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
    except (KeyError, TypeError, ValueError) as e:
        # Return zeros if required columns missing
        return signals
    
    # Handle any NaN values in price data
    close = np.nan_to_num(close, nan=0.0)
    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    volume = np.nan_to_num(volume, nan=0.0)
    
    # Ensure no zero or negative prices
    close = np.where(close <= 0, 1.0, close)
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all technical indicators
    ema_trend = calculate_ema(close, TREND_EMA)
    bb_upper, bb_middle, bb_lower, bb_bandwidth = calculate_bollinger_bands(close, BB_PERIOD, BB_STD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    momentum = calculate_momentum(close, MOMENTUM_PERIOD)
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    adx = calculate_adx(high, low, close, ADX_PERIOD)
    rsi = calculate_rsi(close, 14)
    
    # Calculate bandwidth percentile for squeeze detection
    # Use rolling percentile of bandwidth over last 100 bars
    bandwidth_percentile = np.zeros(n, dtype=np.float64)
    lookback_pct = 100
    for i in range(lookback_pct - 1, n):
        window = bb_bandwidth[i - lookback_pct + 1:i + 1]
        sorted_window = np.sort(window)
        rank = np.searchsorted(sorted_window, bb_bandwidth[i])
        bandwidth_percentile[i] = rank / len(sorted_window)
    
    # Determine minimum valid index (need enough data for all indicators)
    min_valid_index = max(
        TREND_EMA,
        BB_PERIOD + BB_PERIOD,  # Need extra for ADX calculation
        ATR_PERIOD + 1,
        MOMENTUM_PERIOD,
        VOLUME_LOOKBACK,
        ADX_PERIOD * 2,
        lookback_pct
    )
    
    # Generate signals
    for i in range(min_valid_index, n):
        # Skip if any required data is invalid
        if close[i] <= 0 or atr[i] <= 0 or ema_trend[i] <= 0:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 200 EMA
        trend_bullish = close[i] > ema_trend[i] * 1.002  # Small buffer
        trend_bearish = close[i] < ema_trend[i] * 0.998
        
        # Volatility squeeze detection (low bandwidth percentile)
        is_squeeze = bandwidth_percentile[i] < SQUEEZE_THRESHOLD
        
        # Breakout detection
        breakout_long = close[i] > bb_upper[i]
        breakout_short = close[i] < bb_lower[i]
        
        # Volume confirmation
        volume_confirmed = volume_ratio[i] >= VOLUME_THRESHOLD
        
        # Momentum confirmation
        momentum_strong_long = momentum[i] > 0.5  # Positive momentum
        momentum_strong_short = momentum[i] < -0.5  # Negative momentum
        
        # ADX filter (trending market, not choppy)
        trend_market = adx[i] >= ADX_MIN
        
        # RSI filter (avoid overbought/oversold extremes for entries)
        rsi_ok_long = rsi[i] < 75  # Not extremely overbought
        rsi_ok_short = rsi[i] > 25  # Not extremely oversold
        
        # Calculate base signal strength
        signal_strength = 0.0
        
        # Long signal conditions
        if breakout_long and trend_bullish and volume_confirmed and momentum_strong_long and rsi_ok_long:
            # Stronger signal if coming from squeeze
            if is_squeeze:
                signal_strength = 0.8
            else:
                signal_strength = 0.5
            
            # Add momentum component
            signal_strength += min(abs(momentum[i]) / 5.0, 0.2)
        
        # Short signal conditions
        elif breakout_short and trend_bearish and volume_confirmed and momentum_strong_short and rsi_ok_short:
            # Stronger signal if coming from squeeze
            if is_squeeze:
                signal_strength = -0.8
            else:
                signal_strength = -0.5
            
            # Add momentum component
            signal_strength -= min(abs(momentum[i]) / 5.0, 0.2)
        
        # Apply ADX filter (reduce signal in choppy markets)
        if not trend_market:
            signal_strength *= 0.3
        
        # Volatility adjustment (reduce position in very high volatility)
        atr_pct = atr[i] / close[i]
        vol_factor = 1.0
        if atr_pct > 0:
            # Typical 1h ATR% is 0.5-2%, scale inversely
            vol_factor = min(1.0, 0.02 / max(atr_pct, 0.001))
        
        signal_strength *= vol_factor
        
        # Apply minimum signal threshold
        if abs(signal_strength) < MIN_SIGNAL:
            signal_strength = 0.0
        
        # Clip to [-1, 1]
        signal_strength = np.clip(signal_strength, -1.0, 1.0)
        
        signals[i] = signal_strength
    
    return signals