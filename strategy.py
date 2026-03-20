#!/usr/bin/env python3
"""
strategy.py - Trend Volatility Hybrid V13
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Improved trend-following with volatility-based risk management:
    - Primary signal: EMA crossover trend (13/48 EMA) on 4h timeframe
    - Trend filter: Price above/below 200 EMA for major trend direction
    - Volatility filter: ATR-based position sizing to normalize risk
    - Momentum confirmation: RSI in favorable zone (not extreme)
    - Drawdown control: Reduce signal strength during high volatility
    
    Why 4h timeframe:
    - Experiment #004 showed 4h has good Sharpe (0.628)
    - Fewer whipsaws than lower timeframes
    - Better risk/reward for trend following
    - Lower transaction costs relative to signal frequency
    
    Improvements over v12:
    - Simpler signal generation (less overfitting)
    - Better volatility-based signal scaling
    - More conservative entry thresholds
    - Smoother signal transitions

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

name = "trend_volatility_hybrid_v13"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for better drawdown control

# EMA configuration for trend detection
EMA_FAST = 13
EMA_SLOW = 48
EMA_MAJOR = 200

# RSI configuration for entry timing
RSI_PERIOD = 14
RSI_LONG_MIN = 40  # RSI must be above this for longs
RSI_SHORT_MAX = 60  # RSI must be below this for shorts

# Volatility configuration
ATR_PERIOD = 14
VOLATILITY_TARGET = 0.020  # Target ATR as % of price
VOLATILITY_MIN = 0.005  # Minimum ATR % to trade
VOLATILITY_MAX = 0.080  # Maximum ATR % to trade

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 0.80  # Maximum signal magnitude
SMOOTHING_FACTOR = 0.40  # EMA smoothing for signals (0=none, 1=max)
TREND_STRENGTH_MIN = 0.003  # Minimum EMA diff to consider trend valid

# Funding rate configuration (contrarian overlay)
FUNDING_EXTREME = 0.0010  # 0.10% per 8hr
FUNDING_WEIGHT = 0.25  # How much funding affects signal
FUNDING_LOOKBACK = 80


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_ema(close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average using only past data.
    """
    n = len(close)
    if n < period:
        return np.zeros(n, dtype=np.float64)
    
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


def calculate_funding_signal(funding_rate: np.ndarray, lookback: int = 80) -> np.ndarray:
    """
    Calculate funding rate contrarian signal.
    Extreme positive funding → short bias (negative signal)
    Extreme negative funding → long bias (positive signal)
    Returns value in [-FUNDING_WEIGHT, FUNDING_WEIGHT].
    Only uses current/past funding rate (no look-ahead).
    """
    n = len(funding_rate)
    signal = np.zeros(n, dtype=np.float64)
    
    if n < lookback:
        return signal
    
    funding_series = pd.Series(funding_rate)
    rolling_mean = funding_series.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = funding_series.rolling(window=lookback, min_periods=lookback).std()
    
    for i in range(lookback, n):
        fr = funding_rate[i]
        fr_mean = rolling_mean.iloc[i]
        fr_std = rolling_std.iloc[i]
        
        if fr_std > 0 and not np.isnan(fr_std):
            z_score = (fr - fr_mean) / fr_std
            
            # Contrarian: extreme positive funding → short bias
            if z_score > 1.5:
                signal[i] = -FUNDING_WEIGHT * min(1.0, z_score / 3.0)
            elif z_score < -1.5:
                signal[i] = FUNDING_WEIGHT * min(1.0, abs(z_score) / 3.0)
    
    return signal


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Trend Volatility Hybrid V13 Strategy.
    
    Signal Logic:
    1. Calculate EMA trend (13/48 crossover with 200 EMA filter)
    2. Calculate RSI momentum confirmation
    3. Calculate ATR for volatility normalization
    4. Calculate funding rate contrarian overlay
    5. Combine signals with volatility-based scaling
    6. Smooth signals to reduce whipsaws
    7. Apply magnitude filters
    
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
    funding_signal = calculate_funding_signal(funding_rate, FUNDING_LOOKBACK)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = max(
        EMA_MAJOR,
        EMA_SLOW + 5,
        RSI_PERIOD + 1,
        ATR_PERIOD + 1,
        FUNDING_LOOKBACK
    )
    
    # Generate signals
    prev_signal = 0.0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0 or ema_major[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate ATR as percentage of price
        atr_pct = atr[i] / close[i]
        
        # Volatility filter (not too low, not too high)
        if atr_pct < VOLATILITY_MIN or atr_pct > VOLATILITY_MAX:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Calculate EMA trend direction
        ema_diff = (ema_fast[i] - ema_slow[i]) / close[i]
        ema_direction = np.sign(ema_diff)
        ema_strength = abs(ema_diff)
        
        # Check trend strength (avoid weak trends)
        if ema_strength < TREND_STRENGTH_MIN:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Major trend filter (price vs 200 EMA)
        major_direction = np.sign(close[i] - ema_major[i])
        
        # Only trade in direction of major trend
        if ema_direction != major_direction:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # RSI momentum confirmation
        rsi_factor = 1.0
        if ema_direction > 0:
            # Long: want RSI above minimum but not overbought
            if rsi[i] < RSI_LONG_MIN or rsi[i] > 75:
                rsi_factor = 0.0
            elif rsi[i] > 65:
                rsi_factor = 0.6
        elif ema_direction < 0:
            # Short: want RSI below maximum but not oversold
            if rsi[i] > RSI_SHORT_MAX or rsi[i] < 25:
                rsi_factor = 0.0
            elif rsi[i] < 35:
                rsi_factor = 0.6
        
        if rsi_factor == 0.0:
            signals[i] = 0.0
            prev_signal = 0.0
            continue
        
        # Base trend signal strength
        trend_strength = min(1.0, ema_strength / 0.02)  # Normalize to max 1.0
        base_signal = ema_direction * trend_strength * rsi_factor
        
        # Add funding contrarian overlay
        fund_sig = funding_signal[i]
        
        # Combine signals: if funding conflicts with trend, reduce strength
        if np.sign(base_signal) != np.sign(fund_sig) and abs(fund_sig) > 0.1:
            raw_signal = base_signal * 0.75 + fund_sig * 0.25
        else:
            raw_signal = base_signal * 0.85 + fund_sig * 0.15
        
        # Volatility-based signal scaling
        # Reduce signal during high volatility periods (drawdown control)
        vol_factor = VOLATILITY_TARGET / max(atr_pct, 0.001)
        vol_factor = np.clip(vol_factor, 0.4, 1.5)  # Cap reduction during high vol
        raw_signal *= vol_factor
        
        # Signal smoothing (EMA on signals to reduce whipsaws)
        smoothed_signal = SMOOTHING_FACTOR * prev_signal + (1.0 - SMOOTHING_FACTOR) * raw_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
    
    return signals