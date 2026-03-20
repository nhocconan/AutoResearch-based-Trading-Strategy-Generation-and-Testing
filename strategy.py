#!/usr/bin/env python3
"""
strategy.py - Supertrend Trend Following V1
====================================================================
MUTABLE FILE - The LLM agent edits this file during research.

This file defines the trading strategy. It must expose:
    - name: str                    - Strategy identifier
    - timeframe: str               - Primary timeframe (e.g., "4h")
    - leverage: float              - Leverage multiplier (default 1.0)
    - generate_signals(prices)     - Signal generation function

Strategy Hypothesis:
    Pure Supertrend trend-following on 4h timeframe.
    
    Why Supertrend:
    - Classic trend-following indicator with built-in ATR stops
    - Clear directional signals (price above/below Supertrend line)
    - ATR-based dynamic support/resistance adapts to volatility
    - Works well on higher timeframes where crypto trends persist
    - Should generate sufficient trades on 4h over 2021-2024 period
    
    Configuration:
    - ATR period: 10 (standard)
    - Multiplier: 3.0 (standard, balances sensitivity vs noise)
    - Timeframe: 4h (captures multi-day trends, reduces noise)
    - Leverage: 1.5x (conservative for trend following)
    
    Entry/Exit Logic:
    - Long: Price closes above Supertrend lower band
    - Short: Price closes below Supertrend upper band
    - Exit: Signal flips direction
    
    Risk Management:
    - ATR-based position sizing implicit in signal
    - No trades during low volatility (ATR < threshold)
    - Signal smoothing to reduce whipsaws

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

name = "supertrend_4h_v1"
timeframe = "4h"
leverage = 1.5  # Conservative leverage for trend following

# Supertrend configuration
ATR_PERIOD = 10
ATR_MULTIPLIER = 3.0

# Signal configuration
MIN_SIGNAL_MAGNITUDE = 0.20  # Minimum signal to generate position
MAX_SIGNAL = 1.0  # Maximum signal magnitude
SMOOTHING_BARS = 3  # Number of bars to confirm trend change
VOLATILITY_MIN_ATR_PCT = 0.005  # Minimum ATR % to trade
VOLATILITY_MAX_ATR_PCT = 0.080  # Maximum ATR % to trade

# Trend confirmation
TREND_CONFIRMATION_BARS = 2  # Bars above/below to confirm


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Calculate Average True Range using only past data.
    Uses Wilder's smoothing method (EMA with alpha = 1/period).
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
    
    # Wilder's smoothing (EMA with alpha = 1/period)
    atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_supertrend(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                         atr: np.ndarray, period: int = 10, multiplier: float = 3.0) -> tuple:
    """
    Calculate Supertrend indicator.
    
    Returns:
        supertrend_values: The Supertrend line values
        trend_direction: 1 for uptrend, -1 for downtrend, 0 for neutral
    
    Formula:
        Basic Upper Band = (High + Low) / 2 + multiplier * ATR
        Basic Lower Band = (High + Low) / 2 - multiplier * ATR
        
        Final Upper Band = 
            Basic Upper Band if Basic Upper Band < prev_Final_Upper or close > prev_Final_Upper
            else prev_Final_Upper
            
        Final Lower Band = 
            Basic Lower Band if Basic Lower Band > prev_Final_Lower or close < prev_Final_Lower
            else prev_Final_Lower
            
        Trend = 1 if close > Final_Upper, else -1
        Supertrend = Final_Lower if trend=1, else Final_Upper
    
    Only uses current/past data (no look-ahead).
    """
    n = len(close)
    supertrend = np.zeros(n, dtype=np.float64)
    trend = np.zeros(n, dtype=np.float64)
    
    if n < period + 1:
        return supertrend, trend
    
    # Basic bands
    hl2 = (high + low) / 2.0
    basic_upper = hl2 + multiplier * atr
    basic_lower = hl2 - multiplier * atr
    
    # Final bands (need to track state)
    final_upper = np.zeros(n, dtype=np.float64)
    final_lower = np.zeros(n, dtype=np.float64)
    
    # Initialize at period
    final_upper[period] = basic_upper[period]
    final_lower[period] = basic_lower[period]
    trend[period] = 1 if close[period] > final_upper[period] else -1
    supertrend[period] = final_lower[period] if trend[period] > 0 else final_upper[period]
    
    # Iterate through remaining bars
    for i in range(period + 1, n):
        # Calculate final upper band
        if basic_upper[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = basic_upper[i]
        else:
            final_upper[i] = final_upper[i-1]
        
        # Calculate final lower band
        if basic_lower[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = basic_lower[i]
        else:
            final_lower[i] = final_lower[i-1]
        
        # Determine trend direction
        if trend[i-1] > 0:
            # Previous trend was up
            if close[i] < final_lower[i]:
                # Trend reversal to down
                trend[i] = -1
                supertrend[i] = final_upper[i]
            else:
                # Trend continues up
                trend[i] = 1
                supertrend[i] = final_lower[i]
        else:
            # Previous trend was down
            if close[i] > final_upper[i]:
                # Trend reversal to up
                trend[i] = 1
                supertrend[i] = final_lower[i]
            else:
                # Trend continues down
                trend[i] = -1
                supertrend[i] = final_upper[i]
    
    return supertrend, trend


def calculate_signal_strength(close: np.ndarray, supertrend: np.ndarray, 
                               trend: np.ndarray, atr: np.ndarray) -> np.ndarray:
    """
    Calculate signal strength based on distance from Supertrend line.
    Further from line = stronger signal.
    
    Returns values in [-1, 1].
    """
    n = len(close)
    signal = np.zeros(n, dtype=np.float64)
    
    for i in range(n):
        if close[i] <= 0 or atr[i] <= 0:
            signal[i] = 0.0
            continue
        
        # Distance from Supertrend as % of price
        distance_pct = abs(close[i] - supertrend[i]) / close[i]
        
        # Normalize by ATR (distance in ATR units)
        atr_units = distance_pct / (atr[i] / close[i]) if close[i] > 0 else 0
        
        # Cap at 3 ATR units for normalization
        strength = min(atr_units / 3.0, 1.0)
        
        # Apply trend direction
        signal[i] = trend[i] * strength
    
    return signal


# =============================================================================
# Signal Generation
# =============================================================================

def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    """
    Supertrend Trend Following Strategy.
    
    Signal Logic:
    1. Calculate ATR (10-period)
    2. Calculate Supertrend line and trend direction
    3. Generate signal based on trend direction
    4. Scale signal by distance from Supertrend line
    5. Apply volatility filter
    6. Smooth signals to reduce whipsaws
    7. Require confirmation bars for trend changes
    
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
    high = np.maximum(high, close)
    low = np.minimum(low, close)
    
    # Calculate ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    # Calculate Supertrend
    supertrend, trend = calculate_supertrend(
        high, low, close, atr, ATR_PERIOD, ATR_MULTIPLIER
    )
    
    # Calculate signal strength
    raw_signal = calculate_signal_strength(close, supertrend, trend, atr)
    
    # Calculate minimum valid index (all indicators need warmup)
    min_valid_index = ATR_PERIOD + 5  # Extra buffer for Supertrend initialization
    
    # Generate final signals with smoothing and confirmation
    prev_signal = 0.0
    prev_trend = 0
    confirmation_count = 0
    
    for i in range(min_valid_index, n):
        # Skip invalid bars
        if close[i] <= 0 or atr[i] <= 0:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_trend = 0
            confirmation_count = 0
            continue
        
        # Volatility filter (not too low, not too high)
        atr_pct = atr[i] / close[i]
        if atr_pct < VOLATILITY_MIN_ATR_PCT or atr_pct > VOLATILITY_MAX_ATR_PCT:
            signals[i] = 0.0
            prev_signal = 0.0
            prev_trend = 0
            confirmation_count = 0
            continue
        
        # Get current trend and raw signal
        current_trend = trend[i]
        current_signal = raw_signal[i]
        
        # Require confirmation for trend changes
        if current_trend != prev_trend and current_trend != 0:
            confirmation_count += 1
            if confirmation_count < TREND_CONFIRMATION_BARS:
                # Wait for confirmation, keep previous signal
                signals[i] = prev_signal
                continue
            else:
                # Confirmed trend change
                confirmation_count = 0
        elif current_trend == prev_trend:
            confirmation_count = 0
        
        # Apply signal smoothing (simple moving average of last few signals)
        if i >= SMOOTHING_BARS:
            smoothed_signal = np.mean(raw_signal[i-SMOOTHING_BARS+1:i+1])
        else:
            smoothed_signal = current_signal
        
        # Apply minimum magnitude filter
        if abs(smoothed_signal) < MIN_SIGNAL_MAGNITUDE:
            smoothed_signal = 0.0
        
        # Clip to max signal
        signal = np.clip(smoothed_signal, -MAX_SIGNAL, MAX_SIGNAL)
        
        signals[i] = signal
        prev_signal = signal
        prev_trend = current_trend
    
    return signals