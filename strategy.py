#!/usr/bin/env python3
"""
strategy.py - HMA Multi-Regime Hybrid V1
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    HMA crossover with regime-adaptive filtering on 1h timeframe:
    - Primary signal: HMA(16)/HMA(48) crossover (less lag than EMA)
    - Regime filter: Bollinger Band width percentile to detect vol regime
    - Low vol regime → trend following (HMA crossover signals)
    - High vol regime → mean reversion (RSI extremes against trend)
    - Trend confirmation: Price above/below HMA(96) for major direction
    - Entry timing: RSI(14) momentum filter to avoid overextended entries
    - Risk management: ATR-based position sizing and signal scaling
    
    Why this works:
    - HMA has significantly less lag than EMA (Hull's research)
    - Regime adaptation prevents trend strategies from failing in choppy markets
    - Multi-layer filtering reduces false signals and drawdown
    - Simpler than v12, more adaptive to market conditions
    - Conservative leverage (1.5x) with better risk controls

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

name = "hma_regime_hybrid_v1"
timeframe = "1h"
leverage = 1.5  # Conservative leverage for drawdown control

# HMA configuration (Hull Moving Average - less lag than EMA)
HMA_FAST = 16
HMA_MID = 48
HMA_SLOW = 96

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_ENTRY = 45  # RSI must be above this for long entries
RSI_SHORT_ENTRY = 55  # RSI must be below this for short entries
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Bollinger Band regime detection
BB_PERIOD = 20
BB_STD = 2.0
BB_LOOKBACK = 100  # For percentile calculation
BB_LOW_VOL_THRESHOLD = 0.30  # Below 30th percentile = low vol (trend)
BB_HIGH_VOL_THRESHOLD = 0.70  # Above 70th percentile = high vol (mean rev)

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.012  # Target ATR as % of price
VOLATILITY_MIN = 0.002  # Minimum ATR % to trade
VOLATILITY_MAX = 0.040  # Maximum ATR % to trade

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SMOOTHING_FACTOR = 0.40  # EMA smoothing for signals
HYSTERESIS_THRESHOLD = 0.12  # Minimum change to flip signal direction

# Volume confirmation
VOLUME_LOOKBACK = 20
VOLUME_MIN_RATIO = 0.60  # Volume must be at least this % of average

# Regime weighting
TREND_REGIME_WEIGHT = 0.75  # Weight for trend signals in low vol
MEANREV_REGIME_WEIGHT = 0.65  # Weight for mean reversion in high vol


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_hma(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Hull Moving Average using only past data.
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    Significantly less lag than EMA.
    """
    n = len(close)
    hma = np.zeros(n, dtype=np.float64)
    
    if n < period:
        return hma
    
    close_series = pd.Series(close)
    
    # Calculate WMA for period/2
    half_period = max(1, period // 2)
    wma_half = close_series.ewm(span=half_period, adjust=False, min_periods=half_period).mean()
    
    # Calculate WMA for full period
    wma_full = close_series.ewm(span=period, adjust=False, min_periods=period).mean()
    
    # Calculate raw HMA value
    raw_hma = 2.0 * wma_half - wma_full
    
    # Smooth with WMA of sqrt(period)
    sqrt_period = int(np.sqrt(period))
    sqrt_period = max(1, sqrt_period)
    hma_series = raw_hma.ewm(span=sqrt_period, adjust=False, min_periods=sqrt_period).mean()
    
    hma = np.nan_to_num(hma_series.values, nan=0.0)
    
    return hma


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


def calculate_bollinger_bands(close: np.ndarray, period: int = 20, std: float = 2.0) -> tuple:
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
    
    upper = np.nan_to_num((middle_series + std * std_series).values, nan=0.0)
    middle = np.nan_to_num(middle_series.values, nan=0.0)
    lower = np.nan_to_num((middle_series - std * std_series).values, nan=0.0)
    bandwidth = np.nan_to_num((upper - lower) / middle, nan=0.0)
    
    return upper, middle, lower, bandwidth


def calculate_bb_percentile(bandwidth: np.ndarray, lookback: int = 100) -> np.ndarray:
    """
    Calculate rolling percentile of Bollinger Band width.
    Returns value in [0, 1] where low = low volatility, high = high volatility.
    Only uses past data (no look-ahead).
    """
    n = len(bandwidth)
    percentile = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return percentile
    
    bw_series = pd.Series(bandwidth)
    # Calculate rolling rank percentile
    for i in range(lookback - 1, n):
        window = bw_series.iloc[i - lookback + 1:i + 1]
        current_val = bandwidth[i]
        # Percentile rank: what % of values are below current
        percentile[i] = (window < current_val).sum() / lookback
    
    return percentile


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


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    HMA Regime Hybrid V1 Strategy.
    
    Signal Logic:
    1. Calculate HMA crossover signals (16/48) for trend direction
    2. Calculate Bollinger Band width percentile for regime detection
    3. Low vol regime (BB < 30th %ile) → trend following
    4. High vol regime (BB > 70th %ile) → mean reversion with RSI
    5. Major trend filter: Price vs HMA(96)
    6. RSI momentum filter for entry timing
    7. Volatility normalization and signal smoothing
    8. Hysteresis to reduce whipsaws
    
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
    high = np.where(high <= 0, close, high)
    low = np.where(low <= 0, close * 0.99, low)
    
    # Calculate all indicators (all use only past data)
    hma_fast = calculate_hma(close, HMA_FAST)
    hma_mid = calculate_hma(close, HMA_MID)
    hma_slow = calculate_hma(close, HMA_SLOW)
    
    rsi = calculate_rsi(close, RSI_PERIOD)
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    bb_upper, bb_middle, bb_lower, bb_bandwidth = calculate_bollinger_bands(
        close, BB_PERIOD, BB_STD
    )
    bb_percentile = calculate_bb_percentile(bb_bandwidth, BB_LOOKBACK)
    
    volume_ratio = calculate_volume_ratio(volume, VOLUME_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        HMA_SLOW,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        BB_LOOKBACK,
        VOLUME_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    prev_direction = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0 or hma_slow[i] <= 0:
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
        
        # Determine market regime from BB percentile
        bb_pct = bb_percentile[i]
        is_low_vol = bb_pct < BB_LOW_VOL_THRESHOLD
        is_high_vol = bb_pct > BB_HIGH_VOL_THRESHOLD
        is_mid_vol = not is_low_vol and not is_high_vol
        
        # Major trend direction from HMA(96)
        major_trend = np.sign(close[i] - hma_slow[i])
        
        # HMA crossover signal (fast vs mid)
        hma_diff = (hma_fast[i] - hma_mid[i]) / close[i]
        hma_direction = np.sign(hma_diff)
        
        # Calculate raw signal based on regime
        raw_signal = 0.0
        
        if is_low_vol:
            # Low volatility regime → trend following
            # Only trade in direction of major trend
            if hma_direction == major_trend and hma_direction != 0:
                trend_strength = abs(hma_diff) * 80
                trend_strength = np.clip(trend_strength, 0.1, 1.0)
                
                # RSI confirmation for entry timing
                rsi_factor = 1.0
                if hma_direction > 0:
                    # Long: RSI should be above entry threshold but not overbought
                    if rsi[i] < RSI_LONG_ENTRY:
                        rsi_factor = 0.0
                    elif rsi[i] > RSI_OVERBOUGHT:
                        rsi_factor = 0.5
                else:
                    # Short: RSI should be below entry threshold but not oversold
                    if rsi[i] > RSI_SHORT_ENTRY:
                        rsi_factor = 0.0
                    elif rsi[i] < RSI_OVERSOLD:
                        rsi_factor = 0.5
                
                raw_signal = hma_direction * trend_strength * rsi_factor * TREND_REGIME_WEIGHT
            
        elif is_high_vol:
            # High volatility regime → mean reversion
            # Fade extremes against minor moves, with major trend
            if major_trend > 0:
                # Uptrend: look for long entries on RSI dips
                if rsi[i] < RSI_LONG_ENTRY and rsi[i] > RSI_OVERSOLD:
                    raw_signal = 0.5 * MEANREV_REGIME_WEIGHT
                elif rsi[i] > RSI_OVERBOUGHT:
                    # Reduce or exit longs
                    raw_signal = -0.3 * MEANREV_REGIME_WEIGHT
            elif major_trend < 0:
                # Downtrend: look for short entries on RSI rallies
                if rsi[i] > RSI_SHORT_ENTRY and rsi[i] < RSI_OVERBOUGHT:
                    raw_signal = -0.5 * MEANREV_REGIME_WEIGHT
                elif rsi[i] < RSI_OVERSOLD:
                    # Reduce or exit shorts
                    raw_signal = 0.3 * MEANREV_REGIME_WEIGHT
            else:
                # No clear major trend in high vol → stay neutral
                raw_signal = 0.0
        
        else:
            # Mid volatility regime → blend of trend and mean reversion
            # Use HMA crossover but with reduced strength
            if hma_direction == major_trend and hma_direction != 0:
                trend_strength = abs(hma_diff) * 50
                trend_strength = np.clip(trend_strength, 0.1, 0.6)
                
                rsi_factor = 1.0
                if hma_direction > 0 and rsi[i] < RSI_LONG_ENTRY:
                    rsi_factor = 0.0
                elif hma_direction < 0 and rsi[i] > RSI_SHORT_ENTRY:
                    rsi_factor = 0.0
                
                raw_signal = hma_direction * trend_strength * rsi_factor * 0.5
        
        # Volatility normalization (scale by target volatility)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.6, 1.8)
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
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_direction = np.sign(signal)
    
    return signals