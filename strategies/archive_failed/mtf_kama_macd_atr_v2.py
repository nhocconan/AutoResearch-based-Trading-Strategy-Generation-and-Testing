#!/usr/bin/env python3
"""
EXPERIMENT #007 - 4h KAMA Trend + 1h MACD Entry + ATR Volatility Filter
=========================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market efficiency,
providing less lag during trends and less noise during chop. Combined with
MACD histogram for entry timing and ATR volatility filter, this should improve
risk-adjusted returns over simple MA crossovers.

Key differences from current best:
- KAMA(10) instead of HMA for trend (adaptive to market efficiency)
- MACD histogram cross for entry timing (momentum confirmation)
- ATR percentile filter to avoid extreme volatility regimes
- Multi-timeframe: 4h KAMA trend + 1h MACD entries

Why this might beat Sharpe=2.931:
- KAMA reduces whipsaw in choppy markets better than HMA
- MACD histogram provides clearer momentum signals than RSI
- ATR percentile filter avoids trading during panic/euphoria extremes
"""

import numpy as np
import pandas as pd

name = "mtf_kama_macd_atr_v2"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market efficiency - moves faster in trends, slower in chop
    """
    n = len(close)
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 0:
            er[i] = signal / noise
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # Initialize KAMA
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    
    # Calculate EMAs
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    macd_signal = pd.Series(macd_line).ewm(span=signal, min_periods=signal).mean().values
    macd_hist = macd_line - macd_signal
    
    return macd_line, macd_signal, macd_hist


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


def calculate_atr_percentile(atr, close, lookback=100):
    """Calculate ATR as percentile of recent ATR values (volatility regime)"""
    n = len(close)
    atr_pct = np.zeros(n)
    
    # ATR as % of price
    atr_ratio = atr / close
    
    for i in range(lookback - 1, n):
        window = atr_ratio[i - lookback + 1:i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) > 0:
            atr_pct[i] = np.sum(valid <= atr_ratio[i]) / len(valid)
    
    return atr_pct


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    direction = np.ones(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        hl2 = (high[i] + low[i]) / 2
        upper_band = hl2 + multiplier * atr[i]
        lower_band = hl2 - multiplier * atr[i]
        
        if direction[i - 1] == 1:
            if close[i] < lower_band:
                direction[i] = -1
                supertrend[i] = upper_band
            else:
                direction[i] = 1
                supertrend[i] = lower_band
        else:
            if close[i] > upper_band:
                direction[i] = 1
                supertrend[i] = lower_band
            else:
                direction[i] = -1
                supertrend[i] = upper_band
    
    return supertrend, direction


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and risk
    macd_line_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    atr_1h = calculate_atr(high, low, close, period=14)
    atr_pct_1h = calculate_atr_percentile(atr_1h, close, lookback=100)
    
    # 4h KAMA for trend filter (resample 1h → 4h)
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
    
    c_4h = df_4h['close'].values
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    
    # Calculate 4h KAMA
    kama_4h = calculate_kama(c_4h, period=10, fast_period=2, slow_period=30)
    
    # Calculate 4h trend direction based on KAMA slope and price position
    trend_4h = np.zeros(len(c_4h))
    for i in range(20, len(c_4h)):
        if kama_4h[i] > 0 and kama_4h[i-1] > 0:
            kama_slope = (kama_4h[i] - kama_4h[i-5]) / kama_4h[i-5] if kama_4h[i-5] > 0 else 0
            price_vs_kama = (c_4h[i] - kama_4h[i]) / kama_4h[i] if kama_4h[i] > 0 else 0
            
            if kama_slope > 0.001 and price_vs_kama > 0.005:
                trend_4h[i] = 1  # Bullish
            elif kama_slope < -0.001 and price_vs_kama < -0.005:
                trend_4h[i] = -1  # Bearish
    
    # Map 4h trend back to 1h timeframe
    trend_1h = np.zeros(n)
    idx_1h_to_4h = np.arange(n) // 4
    
    for i in range(n):
        idx_4h = idx_1h_to_4h[i]
        if idx_4h < len(trend_4h):
            trend_1h[i] = trend_4h[idx_4h]
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels to reduce churn
    SIZE_FULL = 0.35   # Full position in good conditions
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    
    # MACD histogram thresholds for entry timing
    MACD_LONG_THRESHOLD = 0.0    # MACD hist crosses above 0
    MACD_SHORT_THRESHOLD = 0.0   # MACD hist crosses below 0
    
    # ATR percentile thresholds for volatility filter
    ATR_PCT_MIN = 0.15     # Don't trade in extremely low vol
    ATR_PCT_MAX = 0.85     # Don't trade in extremely high vol
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 20, 14, 100)  # Wait for all indicators
    
    # Track positions for stoploss logic
    position_entry_price = np.zeros(n)
    position_direction = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(macd_hist_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(atr_pct_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        macd_hist = macd_hist_1h[i]
        macd_hist_prev = macd_hist_1h[i-1] if i > 0 else 0
        atr = atr_1h[i]
        atr_pct = atr_pct_1h[i]
        price = close[i]
        
        # Skip if ATR is NaN or zero
        if atr <= 0 or np.isnan(atr):
            signals[i] = 0.0
            continue
        
        # Volatility filter - avoid extreme regimes
        if atr_pct < ATR_PCT_MIN or atr_pct > ATR_PCT_MAX:
            signals[i] = 0.0
            continue
        
        # ATR filter - avoid trading when ATR is extremely high (>5% of price)
        if atr / price > 0.05:
            signals[i] = 0.0
            continue
        
        # Check existing position stoploss first
        if position_direction[i-1] != 0 and i > 0:
            entry_price = position_entry_price[i-1]
            direction = position_direction[i-1]
            
            if direction > 0:  # Long position
                stoploss_price = entry_price - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0  # Stoploss triggered
                    position_direction[i] = 0
                    position_entry_price[i] = 0
                    continue
            elif direction < 0:  # Short position
                stoploss_price = entry_price + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0  # Stoploss triggered
                    position_direction[i] = 0
                    position_entry_price[i] = 0
                    continue
        
        # New entry logic based on trend and MACD
        if trend == 1:  # 4h uptrend
            # Look for MACD histogram crossing above 0 (momentum turning positive)
            if macd_hist > MACD_LONG_THRESHOLD and macd_hist_prev <= MACD_LONG_THRESHOLD:
                signals[i] = SIZE_FULL
                position_entry_price[i] = price
                position_direction[i] = 1
            elif macd_hist > 0 and signals[i-1] > 0:
                # Hold existing long position
                signals[i] = signals[i-1]
                position_entry_price[i] = position_entry_price[i-1]
                position_direction[i] = position_direction[i-1]
            else:
                signals[i] = 0.0
                position_direction[i] = 0
                position_entry_price[i] = 0
        elif trend == -1:  # 4h downtrend
            # Look for MACD histogram crossing below 0 (momentum turning negative)
            if macd_hist < MACD_SHORT_THRESHOLD and macd_hist_prev >= MACD_SHORT_THRESHOLD:
                signals[i] = -SIZE_FULL
                position_entry_price[i] = price
                position_direction[i] = -1
            elif macd_hist < 0 and signals[i-1] < 0:
                # Hold existing short position
                signals[i] = signals[i-1]
                position_entry_price[i] = position_entry_price[i-1]
                position_direction[i] = position_direction[i-1]
            else:
                signals[i] = 0.0
                position_direction[i] = 0
                position_entry_price[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_direction[i] = 0
            position_entry_price[i] = 0
    
    return signals