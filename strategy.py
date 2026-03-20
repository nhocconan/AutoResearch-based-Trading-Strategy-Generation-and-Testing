#!/usr/bin/env python3
"""
strategy.py - Multi-Factor Mean Reversion with Trend Filter
============================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "1h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Combine funding rate mean reversion with trend filter and RSI confirmation.
    - Extreme funding rates suggest overcrowded positions → mean reversion
    - Trade only in direction of longer-term trend (50-period SMA)
    - RSI confirms entry timing (avoid catching falling knives)

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

name = "funding_mean_reversion_trend"
timeframe = "1h"
leverage = 2.0  # Conservative leverage for mean reversion strategy

# Strategy parameters
TREMA_PERIOD = 50           # Trend filter SMA period
RSI_PERIOD = 14             # RSI calculation period
FUNDING_LOOKBACK = 100      # Lookback for funding rate z-score
FUNDING_THRESHOLD = 1.5     # Z-score threshold for extreme funding
RSI_OVERBOUGHT = 65         # RSI level for overbought
RSI_OVERSOLD = 35           # RSI level for oversold
VOLATILITY_WINDOW = 20      # Window for volatility adjustment


# =============================================================================
# Signal Generation
# =============================================================================

def calculate_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate RSI using only past data (no look-ahead).
    
    Args:
        close: Array of close prices
        period: RSI calculation period
    
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
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    
    # Initialize average gain/loss with SMA
    avg_gain = np.zeros(n, dtype=np.float64)
    avg_loss = np.zeros(n, dtype=np.float64)
    
    # First average (simple MA of first 'period' changes)
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])
    
    # Subsequent averages (Wilder's smoothing)
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i-1]) / period
    
    # Calculate RS and RSI
    rs = np.zeros(n, dtype=np.float64)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    # Handle division by zero (all gains, no losses)
    rsi[avg_loss == 0] = 100.0
    
    return rsi


def calculate_funding_zscore(funding_rates: np.ndarray, lookback: int = 100) -> np.ndarray:
    """
    Calculate z-score of funding rates for mean reversion signal.
    Only uses past funding rate data (no look-ahead).
    
    Args:
        funding_rates: Array of funding rates
        lookback: Rolling window for mean/std calculation
    
    Returns:
        Array of z-scores
    """
    n = len(funding_rates)
    zscore = np.zeros(n, dtype=np.float64)
    
    # Use pandas rolling for clean calculation
    funding_series = pd.Series(funding_rates)
    rolling_mean = funding_series.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = funding_series.rolling(window=lookback, min_periods=lookback).std()
    
    # Calculate z-score where we have enough data
    mask = rolling_std > 0
    zscore[mask] = (funding_rates - rolling_mean.values) / rolling_std.values
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Multi-Factor Mean Reversion Strategy with Trend Filter.
    
    Signal Logic:
    1. Calculate funding rate z-score (mean reversion signal)
    2. Calculate trend direction (50-period SMA)
    3. Calculate RSI for entry timing
    4. Combine signals with weights
    
    Entry Conditions:
    - LONG: Funding z-score < -threshold (extreme negative) AND price > trend SMA AND RSI < oversold
    - SHORT: Funding z-score > threshold (extreme positive) AND price < trend SMA AND RSI > overbought
    
    Args:
        prices: DataFrame with columns [open_time, open, high, low, close, volume, funding_rate, ...]
    
    Returns:
        np.ndarray of signals, same length as prices. Values in [-1, 1].
    """
    n = len(prices)
    signals = np.zeros(n, dtype=np.float64)
    
    # Extract close prices
    close = prices["close"].values
    
    # Check if funding_rate column exists
    has_funding = "funding_rate" in prices.columns
    if has_funding:
        funding_rates = prices["funding_rate"].values
    else:
        # If no funding data, create dummy (strategy will rely on price signals only)
        funding_rates = np.zeros(n, dtype=np.float64)
    
    # Calculate trend filter (50-period SMA)
    close_series = pd.Series(close)
    trend_sma = close_series.rolling(window=TREMA_PERIOD, min_periods=TREMA_PERIOD).mean().values
    
    # Calculate RSI
    rsi = calculate_rsi(close, RSI_PERIOD)
    
    # Calculate funding z-score
    funding_zscore = calculate_funding_zscore(funding_rates, FUNDING_LOOKBACK)
    
    # Calculate volatility for position sizing adjustment
    returns = np.diff(close) / close[:-1]
    returns = np.insert(returns, 0, 0.0)  # Align with close array
    vol_series = pd.Series(returns).rolling(window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW).std().values
    vol_series = np.nan_to_num(vol_series, nan=0.0)
    
    # Generate signals
    min_valid_index = max(TREMA_PERIOD, FUNDING_LOOKBACK, RSI_PERIOD + 1)
    
    for i in range(min_valid_index, n):
        # Skip if any required data is NaN
        if np.isnan(trend_sma[i]) or np.isnan(rsi[i]) or np.isnan(funding_zscore[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to SMA
        price_above_trend = close[i] > trend_sma[i]
        price_below_trend = close[i] < trend_sma[i]
        
        # Funding signal: extreme values suggest mean reversion
        funding_extreme_long = funding_zscore[i] < -FUNDING_THRESHOLD  # Very negative funding → long
        funding_extreme_short = funding_zscore[i] > FUNDING_THRESHOLD  # Very positive funding → short
        
        # RSI confirmation
        rsi_oversold = rsi[i] < RSI_OVERSOLD
        rsi_overbought = rsi[i] > RSI_OVERBOUGHT
        
        # Volatility adjustment (reduce position size in high volatility)
        vol_factor = 1.0
        if vol_series[i] > 0:
            # Normalize volatility (assume typical 1h vol ~0.01-0.02)
            vol_factor = min(1.0, 0.015 / max(vol_series[i], 0.001))
        
        # Combine signals
        long_signal = 0.0
        short_signal = 0.0
        
        # Long entry: extreme negative funding + uptrend + oversold RSI
        if funding_extreme_long and price_above_trend and rsi_oversold:
            long_signal = 1.0 * vol_factor
        elif funding_extreme_long and price_above_trend:
            # Weaker signal without RSI confirmation
            long_signal = 0.5 * vol_factor
        elif funding_extreme_long and rsi_oversold:
            # Weaker signal without trend confirmation
            long_signal = 0.5 * vol_factor
        
        # Short entry: extreme positive funding + downtrend + overbought RSI
        if funding_extreme_short and price_below_trend and rsi_overbought:
            short_signal = 1.0 * vol_factor
        elif funding_extreme_short and price_below_trend:
            # Weaker signal without RSI confirmation
            short_signal = 0.5 * vol_factor
        elif funding_extreme_short and rsi_overbought:
            # Weaker signal without trend confirmation
            short_signal = 0.5 * vol_factor
        
        # Net signal (long - short)
        signals[i] = long_signal - short_signal
    
    return signals