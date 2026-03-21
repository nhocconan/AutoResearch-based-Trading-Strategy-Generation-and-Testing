#!/usr/bin/env python3
"""
EXPERIMENT #031 - MTF KAMA+RSI+Supertrend (1h+4h Simplified Vectorized v2)
==================================================================================================
Hypothesis: Previous crashes (#027-#030) were due to complex position state tracking and indexing errors.
This version uses fully vectorized signal generation without mutable state variables.

Key improvements from #030 crash:
- Remove ALL position state tracking (in_position, entry_price, etc.)
- Pure vectorized signal generation
- 1h timeframe (more stable than 15m, less noise)
- 4h trend filter using mtf_data helper (MANDATORY for alignment)
- KAMA for adaptive trend + RSI for pullback + Supertrend for direction
- Position size: 0.30 max (conservative for drawdown control)
- Signal changes trigger exits naturally (no state to track)

Why this should work:
- Vectorized = no indexing crashes
- 1h+4h proven in #025 (Sharpe=0.025, survived 2022 crash)
- KAMA adapts to volatility better than EMA
- Supertrend provides clean stoploss levels
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_rsi_supertrend_1h_4h_v2"
timeframe = "1h"
leverage = 1.0


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.full(n, 50.0)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=period, min_periods=period).mean().values
    avg_loss = pd.Series(loss).rolling(window=period, min_periods=period).mean().values
    
    rsi = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rsi[i] = 100.0
        elif avg_gain[i] == 0:
            rsi[i] = 0.0
        else:
            rsi[i] = 100 - (100 / (1 + avg_gain[i] / avg_loss[i]))
    
    return np.nan_to_num(rsi, nan=50.0)


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator - vectorized"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)  # Default to 1 (bullish)
    
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
    # Initialize
    supertrend[period - 1] = lower_band[period - 1]
    trend_direction[period - 1] = 1
    
    for i in range(period, n):
        if trend_direction[i - 1] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i - 1])
            if close[i] < supertrend[i]:
                supertrend[i] = upper_band[i]
                trend_direction[i] = -1
            else:
                trend_direction[i] = 1
        else:
            supertrend[i] = min(upper_band[i], supertrend[i - 1])
            if close[i] > supertrend[i]:
                supertrend[i] = lower_band[i]
                trend_direction[i] = 1
            else:
                trend_direction[i] = -1
    
    return supertrend, trend_direction


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average - vectorized"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        
        if volatility == 0:
            er = 1.0
        else:
            er = change / volatility
        
        sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_bollinger_bands(close, period=20, std_mult=2.0):
    """Calculate Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    
    # Bandwidth = (upper - lower) / sma
    bw = np.zeros(n)
    for i in range(period - 1, n):
        if sma[i] > 0:
            bw[i] = (upper[i] - lower[i]) / sma[i]
        else:
            bw[i] = 0.0
    
    return upper, lower, bw


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Initialize signals array
    signals = np.zeros(n)
    
    # Calculate 1h indicators
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    supertrend_1h, st_direction_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    kama_1h = calculate_kama(close, period=10)
    _, _, bbw_1h = calculate_bollinger_bands(close, period=20, std_mult=2.0)
    
    # 4h trend filters using mtf_data helper (MANDATORY)
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h indicators on actual 4h data
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    kama_4h = calculate_kama(close_4h, period=10)
    supertrend_4h, st_direction_4h = calculate_supertrend(high_4h, low_4h, close_4h, period=10, multiplier=3.0)
    
    # Align 4h indicators to 1h timeframe (auto shift for completed bars only)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
    
    # Ensure aligned arrays are proper numpy arrays
    kama_4h_aligned = np.asarray(kama_4h_aligned, dtype=np.float64)
    st_direction_4h_aligned = np.asarray(st_direction_4h_aligned, dtype=np.float64)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    SIZE_ZERO = 0.0
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 60
    
    # BBW percentile for regime filter (avoid chop)
    bbw_window = 100
    
    # Minimum bars for valid signals
    first_valid = max(100, int(n * 0.05))
    
    # Vectorized signal generation
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0:
            signals[i] = SIZE_ZERO
            continue
        
        if np.isnan(kama_4h_aligned[i]) or np.isnan(st_direction_4h_aligned[i]):
            signals[i] = SIZE_ZERO
            continue
        
        # Ensure st_direction is valid integer
        st_4h_dir = int(np.sign(st_direction_4h_aligned[i]))
        if st_4h_dir == 0:
            st_4h_dir = 1  # Default to bullish if unclear
        
        # 4h trend direction from KAMA
        trend_4h = 0
        if kama_4h_aligned[i] > 0 and close[i] > kama_4h_aligned[i]:
            trend_4h = 1
        elif kama_4h_aligned[i] > 0 and close[i] < kama_4h_aligned[i]:
            trend_4h = -1
        
        # 1h indicators
        rsi_val = rsi_1h[i]
        st_direction_1h_val = int(st_direction_1h[i])
        bbw = bbw_1h[i]
        
        # BBW regime filter (avoid very low volatility chop)
        bbw_ok = True
        if i >= bbw_window:
            bbw_hist = bbw_1h[max(0, i - bbw_window):i]
            if len(bbw_hist) >= 10:
                bbw_median = np.median(bbw_hist)
                if bbw < bbw_median * 0.6:  # BBW too low = avoid chop
                    bbw_ok = False
        
        if not bbw_ok:
            signals[i] = SIZE_ZERO
            continue
        
        # Entry logic: 4h trend + 1h RSI pullback + 1h Supertrend confirmation
        if trend_4h == 1 and st_4h_dir == 1:  # Bullish trend on 4h
            # Wait for RSI pullback on 1h, confirm with 1h Supertrend
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                st_direction_1h_val == 1):
                signals[i] = SIZE_FULL
            elif rsi_val > RSI_LONG_MAX and st_direction_1h_val == 1:
                # Strong momentum - hold but reduce size
                signals[i] = SIZE_HALF
            elif close[i] < supertrend_1h[i]:
                # Trend broken - exit
                signals[i] = SIZE_ZERO
            else:
                signals[i] = signals[i - 1] if i > 0 else SIZE_ZERO
                
        elif trend_4h == -1 and st_4h_dir == -1:  # Bearish trend on 4h
            # Wait for RSI pullback on 1h, confirm with 1h Supertrend
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                st_direction_1h_val == -1):
                signals[i] = -SIZE_FULL
            elif rsi_val < RSI_SHORT_MIN and st_direction_1h_val == -1:
                # Strong momentum - hold but reduce size
                signals[i] = -SIZE_HALF
            elif close[i] > supertrend_1h[i]:
                # Trend broken - exit
                signals[i] = SIZE_ZERO
            else:
                signals[i] = signals[i - 1] if i > 0 else SIZE_ZERO
        else:
            signals[i] = SIZE_ZERO
    
    return signals