#!/usr/bin/env python3
"""
EXPERIMENT #012 - Supertrend + MACD Histogram + Z-Score Filter
================================================================
Hypothesis: Supertrend provides cleaner trend signals than Donchian with built-in ATR volatility adjustment.
MACD histogram crosses capture momentum shifts earlier than RSI pullbacks. Z-score filter avoids extreme
mean-reversion traps during strong trends.

Key differences from mtf_donchian_rsi_atr_v1:
- Supertrend(10, 3.0) instead of Donchian(20) for trend (volatility-adaptive)
- MACD histogram cross for entries instead of RSI pullback (momentum-based)
- Z-score(20) filter instead of BB Width percentile (statistical extremes)
- Multi-timeframe: 4h Supertrend trend + 1h MACD entries

Why this might beat Sharpe=5.677:
- Supertrend flips less frequently than Donchian breakouts (reduces whipsaw)
- MACD histogram leads price action better than RSI in strong trends
- Z-score provides cleaner statistical filter than BB Width
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_macd_zscore_v1"
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
    atr[period - 1] = np.mean(tr[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """
    Calculate Supertrend indicator
    Returns: supertrend_line, trend_direction (1=bullish, -1=bearish)
    """
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend = np.zeros(n)
    
    # Calculate basic bands
    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # Initialize
    supertrend[0] = upper_band[0]
    trend[0] = -1  # Start bearish
    
    for i in range(1, n):
        if trend[i - 1] == 1:
            # Previous trend was bullish
            if close[i] > lower_band[i]:
                supertrend[i] = lower_band[i]
                trend[i] = 1
            else:
                supertrend[i] = upper_band[i]
                trend[i] = -1
        else:
            # Previous trend was bearish
            if close[i] < upper_band[i]:
                supertrend[i] = upper_band[i]
                trend[i] = -1
            else:
                supertrend[i] = lower_band[i]
                trend[i] = 1
    
    return supertrend, trend


def calculate_macd(close, fast=12, slow=26, signal=9):
    """
    Calculate MACD indicator
    Returns: macd_line, signal_line, histogram
    """
    n = len(close)
    
    # Calculate EMAs
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_zscore(close, period=20):
    """Calculate Z-score for statistical extremes"""
    n = len(close)
    mean = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    std = pd.Series(close).rolling(window=period, min_periods=period).std().values
    
    zscore = np.zeros(n)
    mask = std > 0
    zscore[mask] = (close[mask] - mean[mask]) / std[mask]
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and filter
    zscore_1h = calculate_zscore(close, period=20)
    
    # MACD for momentum entries
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # 4h Supertrend for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    # Resample to 4h
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    
    # Calculate 4h Supertrend
    _, trend_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = min(idx_1h_to_4h[i], len(trend_4h) - 1)
        if idx_4h >= 0 and idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    
    # MACD histogram thresholds for momentum entries
    MACD_LONG_THRESHOLD = 0.0    # Histogram crossing above zero
    MACD_SHORT_THRESHOLD = 0.0   # Histogram crossing below zero
    
    # Z-score filter thresholds (avoid extreme mean-reversion)
    ZSCORE_MAX = 2.0    # Don't long if price is > 2 std above mean
    ZSCORE_MIN = -2.0   # Don't short if price is < 2 std below mean
    
    # ATR for stoploss
    atr_1h = calculate_atr(high, low, close, period=14)
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 26, 20, 10)  # Wait for all indicators
    
    # Track entry prices for stoploss
    entry_price_long = np.zeros(n)
    entry_price_short = np.zeros(n)
    position_active = np.zeros(n)  # 1=long, -1=short, 0=none
    
    for i in range(first_valid, n):
        if np.isnan(zscore_1h[i]) or np.isnan(macd_hist[i]) or np.isnan(trend_1h[i]) or np.isnan(atr_1h[i]):
            signals[i] = 0.0
            position_active[i] = 0
            continue
        
        trend = trend_1h[i]
        macd_h = macd_hist[i]
        macd_h_prev = macd_hist[i - 1] if i > 0 else 0
        zscore_val = zscore_1h[i]
        atr = atr_1h[i]
        price = close[i]
        
        # Z-score filter - avoid extreme statistical positions
        if zscore_val > ZSCORE_MAX:
            # Price too high - don't go long
            if position_active[i - 1] == 1:
                # Check stoploss for existing long
                if i > 0 and entry_price_long[i - 1] > 0:
                    stoploss_price = entry_price_long[i - 1] - ATR_STOP_MULT * atr
                    if price < stoploss_price:
                        signals[i] = 0.0
                        position_active[i] = 0
                    else:
                        signals[i] = signals[i - 1]
                        position_active[i] = 1
                else:
                    signals[i] = 0.0
                    position_active[i] = 0
            else:
                signals[i] = 0.0
                position_active[i] = 0
            continue
        
        if zscore_val < ZSCORE_MIN:
            # Price too low - don't go short
            if position_active[i - 1] == -1:
                # Check stoploss for existing short
                if i > 0 and entry_price_short[i - 1] > 0:
                    stoploss_price = entry_price_short[i - 1] + ATR_STOP_MULT * atr
                    if price > stoploss_price:
                        signals[i] = 0.0
                        position_active[i] = 0
                    else:
                        signals[i] = signals[i - 1]
                        position_active[i] = -1
                else:
                    signals[i] = 0.0
                    position_active[i] = 0
            else:
                signals[i] = 0.0
                position_active[i] = 0
            continue
        
        if trend == 1:  # 4h uptrend - look for long entries
            # MACD histogram crossing above zero (momentum shift)
            if macd_h > MACD_LONG_THRESHOLD and macd_h_prev <= MACD_LONG_THRESHOLD:
                signals[i] = SIZE_FULL
                entry_price_long[i] = price
                position_active[i] = 1
            elif macd_h > 0:
                # MACD positive but no fresh cross - hold or half position
                if position_active[i - 1] == 1:
                    # Check stoploss
                    if i > 0 and entry_price_long[i - 1] > 0:
                        stoploss_price = entry_price_long[i - 1] - ATR_STOP_MULT * atr
                        if price < stoploss_price:
                            signals[i] = 0.0
                            position_active[i] = 0
                        else:
                            signals[i] = SIZE_HALF
                            position_active[i] = 1
                    else:
                        signals[i] = SIZE_HALF
                        position_active[i] = 1
                else:
                    signals[i] = 0.0
                    position_active[i] = 0
            else:
                # MACD negative - check if we need to exit long
                if position_active[i - 1] == 1:
                    if i > 0 and entry_price_long[i - 1] > 0:
                        stoploss_price = entry_price_long[i - 1] - ATR_STOP_MULT * atr
                        if price < stoploss_price:
                            signals[i] = 0.0
                            position_active[i] = 0
                        else:
                            signals[i] = signals[i - 1]
                            position_active[i] = 1
                    else:
                        signals[i] = 0.0
                        position_active[i] = 0
                else:
                    signals[i] = 0.0
                    position_active[i] = 0
        elif trend == -1:  # 4h downtrend - look for short entries
            # MACD histogram crossing below zero (momentum shift)
            if macd_h < MACD_SHORT_THRESHOLD and macd_h_prev >= MACD_SHORT_THRESHOLD:
                signals[i] = -SIZE_FULL
                entry_price_short[i] = price
                position_active[i] = -1
            elif macd_h < 0:
                # MACD negative but no fresh cross - hold or half position
                if position_active[i - 1] == -1:
                    # Check stoploss
                    if i > 0 and entry_price_short[i - 1] > 0:
                        stoploss_price = entry_price_short[i - 1] + ATR_STOP_MULT * atr
                        if price > stoploss_price:
                            signals[i] = 0.0
                            position_active[i] = 0
                        else:
                            signals[i] = -SIZE_HALF
                            position_active[i] = -1
                    else:
                        signals[i] = -SIZE_HALF
                        position_active[i] = -1
                else:
                    signals[i] = 0.0
                    position_active[i] = 0
            else:
                # MACD positive - check if we need to exit short
                if position_active[i - 1] == -1:
                    if i > 0 and entry_price_short[i - 1] > 0:
                        stoploss_price = entry_price_short[i - 1] + ATR_STOP_MULT * atr
                        if price > stoploss_price:
                            signals[i] = 0.0
                            position_active[i] = 0
                        else:
                            signals[i] = signals[i - 1]
                            position_active[i] = -1
                    else:
                        signals[i] = 0.0
                        position_active[i] = 0
                else:
                    signals[i] = 0.0
                    position_active[i] = 0
        else:  # No clear trend (trend=0)
            signals[i] = 0.0
            position_active[i] = 0
        
        # Copy entry prices forward if position still active
        if i > 0:
            if entry_price_long[i] == 0 and position_active[i] == 1:
                entry_price_long[i] = entry_price_long[i - 1]
            if entry_price_short[i] == 0 and position_active[i] == -1:
                entry_price_short[i] = entry_price_short[i - 1]
    
    return signals