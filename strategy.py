#!/usr/bin/env python3
"""
EXPERIMENT #015 - Supertrend + MACD + RSI + ATR Regime Filter
===============================================================================
Hypothesis: Supertrend provides clean trend direction, MACD histogram confirms 
momentum alignment, RSI identifies pullback entries. ATR regime filter avoids 
extreme volatility periods. This combination should reduce whipsaws vs pure 
Supertrend while capturing strong trends.

Key innovations vs previous Supertrend attempts (#008, #010, #011, #013):
- MACD histogram filter ensures momentum aligns with trend (reduces false entries)
- ATR volatility regime filter skips trading when ATR > 4% of price
- Discrete position sizing (0.0, ±0.20, ±0.35) minimizes churn costs
- Proper trailing stop at 2.5*ATR from entry price
- RSI exit thresholds prevent holding through extreme overbought/oversold

Why this might beat Sharpe=2.139:
- Supertrend worked in #011 (Sharpe=1.459) and #013 (Sharpe=1.454)
- Failed in #008, #010 due to no momentum filter and poor position sizing
- Adding MACD histogram should filter out weak trends
- Conservative max position 0.35 limits drawdown during crashes
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_macd_rsi_atr_v1"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period] = np.mean(tr[1:period+1])
    
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator - returns trend direction (1/-1)"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    median_price = (high + low) / 2
    upper_band = median_price + multiplier * atr
    lower_band = median_price - multiplier * atr
    
    supertrend = np.zeros(n)
    trend = np.ones(n)  # 1 = bullish, -1 = bearish
    
    supertrend[period] = upper_band[period]
    
    for i in range(period + 1, n):
        if trend[i - 1] == 1:
            if close[i] < lower_band[i - 1]:
                trend[i] = -1
                supertrend[i] = upper_band[i]
            else:
                trend[i] = 1
                supertrend[i] = lower_band[i]
        else:
            if close[i] > upper_band[i - 1]:
                trend[i] = 1
                supertrend[i] = lower_band[i]
            else:
                trend[i] = -1
                supertrend[i] = upper_band[i]
    
    return trend


def calculate_macd(close, fast=12, slow=26, signal_period=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    
    # EMA calculations
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    
    ema_fast[0] = close[0]
    ema_slow[0] = close[0]
    
    k_fast = 2.0 / (fast + 1)
    k_slow = 2.0 / (slow + 1)
    
    for i in range(1, n):
        ema_fast[i] = k_fast * close[i] + (1 - k_fast) * ema_fast[i - 1]
        ema_slow[i] = k_slow * close[i] + (1 - k_slow) * ema_slow[i - 1]
    
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    signal_line = np.zeros(n)
    signal_line[signal_period] = np.mean(macd_line[:signal_period+1])
    
    k_signal = 2.0 / (signal_period + 1)
    for i in range(signal_period + 1, n):
        signal_line[i] = k_signal * macd_line[i] + (1 - k_signal) * signal_line[i - 1]
    
    histogram = macd_line - signal_line
    
    return histogram


def calculate_rsi(close, period=14):
    """Calculate RSI with proper initialization"""
    n = len(close)
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    # First average
    avg_gain[period] = np.mean(gain[1:period+1])
    avg_loss[period] = np.mean(loss[1:period+1])
    
    # Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    mask = avg_loss > 0
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    
    rsi = np.zeros(n)
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    rsi[avg_loss == 0] = 100
    
    return rsi


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Calculate all indicators
    atr = calculate_atr(high, low, close, period=14)
    supertrend = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    macd_hist = calculate_macd(close, fast=12, slow=26, signal_period=9)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35
    SIZE_HALF = 0.20
    
    # Thresholds
    RSI_LONG_ENTRY = 45
    RSI_SHORT_ENTRY = 55
    RSI_EXIT_LONG = 70
    RSI_EXIT_SHORT = 30
    ATR_STOP_MULT = 2.5
    
    # Wait for all indicators to be valid
    first_valid = 30
    
    # Track positions
    position = np.zeros(n, dtype=int)  # 1 = long, -1 = short, 0 = flat
    entry_price = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for NaN
        if np.isnan(atr[i]) or np.isnan(rsi[i]) or np.isnan(macd_hist[i]):
            signals[i] = 0.0
            position[i] = 0
            continue
        
        price = close[i]
        current_atr = atr[i]
        
        # ATR volatility filter - skip if ATR > 4% of price (too volatile)
        if current_atr / price > 0.04:
            signals[i] = 0.0
            position[i] = 0
            entry_price[i] = 0
            continue
        
        # Check trailing stop for existing positions
        if position[i - 1] != 0:
            prev_pos = position[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            
            if prev_pos == 1:  # Long
                stoploss = prev_entry - ATR_STOP_MULT * current_atr
                if price < stoploss:
                    signals[i] = 0.0
                    position[i] = 0
                    entry_price[i] = 0
                    continue
            elif prev_pos == -1:  # Short
                stoploss = prev_entry + ATR_STOP_MULT * current_atr
                if price > stoploss:
                    signals[i] = 0.0
                    position[i] = 0
                    entry_price[i] = 0
                    continue
        
        # Get current trend and momentum
        trend = supertrend[i]
        macd_momentum = 1 if macd_hist[i] > 0 else -1
        rsi_val = rsi[i]
        
        # LONG entries: uptrend + positive momentum
        if trend == 1 and macd_momentum == 1:
            if rsi_val < RSI_LONG_ENTRY:
                # Strong pullback - full position
                signals[i] = SIZE_FULL
                position[i] = 1
                entry_price[i] = price
            elif rsi_val < 50:
                # Moderate pullback - half position
                signals[i] = SIZE_HALF
                position[i] = 1
                entry_price[i] = price
            elif position[i - 1] == 1:
                # Hold long position
                signals[i] = signals[i - 1]
                position[i] = 1
                entry_price[i] = entry_price[i - 1]
            elif rsi_val > RSI_EXIT_LONG:
                # Exit on overbought
                signals[i] = 0.0
                position[i] = 0
                entry_price[i] = 0
            else:
                signals[i] = 0.0
                position[i] = 0
                entry_price[i] = 0
                
        # SHORT entries: downtrend + negative momentum
        elif trend == -1 and macd_momentum == -1:
            if rsi_val > RSI_SHORT_ENTRY:
                # Strong rally - full short
                signals[i] = -SIZE_FULL
                position[i] = -1
                entry_price[i] = price
            elif rsi_val > 50:
                # Moderate rally - half short
                signals[i] = -SIZE_HALF
                position[i] = -1
                entry_price[i] = price
            elif position[i - 1] == -1:
                # Hold short position
                signals[i] = signals[i - 1]
                position[i] = -1
                entry_price[i] = entry_price[i - 1]
            elif rsi_val < RSI_EXIT_SHORT:
                # Exit on oversold
                signals[i] = 0.0
                position[i] = 0
                entry_price[i] = 0
            else:
                signals[i] = 0.0
                position[i] = 0
                entry_price[i] = 0
        else:
            # No clear trend or momentum mismatch - exit
            signals[i] = 0.0
            position[i] = 0
            entry_price[i] = 0
    
    return signals