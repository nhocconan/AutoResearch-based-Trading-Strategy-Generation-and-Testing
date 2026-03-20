#!/usr/bin/env python3
"""
EXPERIMENT #012 - EMA Crossover + MACD Histogram + ADX Filter + ATR Stop
=========================================================================
Hypothesis: EMA crossover (21/55) provides smoother trend signals than Donchian breakouts.
Combined with MACD histogram crosses for entry timing and ADX strength filter, this should
reduce whipsaw while capturing sustained trends. ATR-based stops manage risk dynamically.

Key differences from mtf_donchian_rsi_atr_v1:
- EMA(21/55) crossover instead of Donchian (smoother, less false breakouts)
- MACD histogram cross for entry timing (momentum confirmation)
- ADX(14) > 25 filter (only trade when trend strength is real)
- Multi-timeframe: 4h EMA trend + 1h MACD entries + ADX filter

Why this might beat Sharpe=0.517:
- EMA crossover filters out noise better than pure price breakouts
- MACD histogram adds momentum confirmation at entry
- ADX filter avoids choppy consolidation periods entirely
- Proven multi-timeframe approach (4h trend + 1h entries)
"""

import numpy as np
import pandas as pd

name = "mtf_ema_macd_adx_v1"
timeframe = "1h"
leverage = 1.0


def calculate_ema(close, period):
    """Calculate EMA with proper smoothing"""
    n = len(close)
    ema = np.zeros(n)
    multiplier = 2.0 / (period + 1)
    
    # Initialize with SMA
    ema[period - 1] = np.mean(close[:period])
    
    for i in range(period, n):
        ema[i] = (close[i] - ema[i - 1]) * multiplier + ema[i - 1]
    
    return ema


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    
    ema_fast = calculate_ema(close, fast)
    ema_slow = calculate_ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    
    # Signal line is EMA of MACD line
    signal_line = np.zeros(n)
    multiplier = 2.0 / (signal + 1)
    
    # Find first valid MACD value
    first_valid = slow - 1
    signal_line[first_valid] = macd_line[first_valid]
    
    for i in range(first_valid + 1, n):
        if not np.isnan(macd_line[i]):
            signal_line[i] = (macd_line[i] - signal_line[i - 1]) * multiplier + signal_line[i - 1]
    
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength"""
    n = len(close)
    
    # Calculate True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # Calculate +DM and -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i - 1]
        low_diff = low[i - 1] - low[i]
        
        if high_diff > low_diff and high_diff > 0:
            plus_dm[i] = high_diff
        if low_diff > high_diff and low_diff > 0:
            minus_dm[i] = low_diff
    
    # Smooth using Wilder's method
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    
    # Initialize with first period average
    atr[period - 1] = np.mean(tr[1:period + 1])
    plus_di[period - 1] = 100 * np.mean(plus_dm[1:period + 1]) / atr[period - 1] if atr[period - 1] > 0 else 0
    minus_di[period - 1] = 100 * np.mean(minus_dm[1:period + 1]) / atr[period - 1] if atr[period - 1] > 0 else 0
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        
        plus_di_smooth = (plus_di[i - 1] * (period - 1) + 100 * plus_dm[i]) / period
        minus_di_smooth = (minus_di[i - 1] * (period - 1) + 100 * minus_dm[i]) / period
        
        plus_di[i] = 100 * plus_di_smooth / atr[i] if atr[i] > 0 else 0
        minus_di[i] = 100 * minus_di_smooth / atr[i] if atr[i] > 0 else 0
    
    # Calculate DX and ADX
    dx = np.zeros(n)
    adx = np.zeros(n)
    
    for i in range(period, n):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum
    
    # Smooth DX to get ADX
    adx[2 * period - 1] = np.mean(dx[period:2 * period])
    
    for i in range(2 * period, n):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    
    return adx


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing and risk
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    adx_1h = calculate_adx(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # 4h EMA trend filter (resample 1h → 4h)
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
    n_4h = len(c_4h)
    
    # Calculate 4h EMA crossover (21/55)
    ema_21_4h = calculate_ema(c_4h, 21)
    ema_55_4h = calculate_ema(c_4h, 55)
    
    # 4h trend direction based on EMA crossover
    trend_4h = np.zeros(n_4h)
    for i in range(55, n_4h):
        if ema_21_4h[i] > ema_55_4h[i]:
            trend_4h[i] = 1  # Bullish
        elif ema_21_4h[i] < ema_55_4h[i]:
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
    SIZE_FULL = 0.35   # Full position in strong trend
    SIZE_HALF = 0.20   # Reduced position in moderate trend
    
    # ADX threshold for trend strength
    ADX_MIN = 25       # Only trade when ADX indicates strong trend
    
    # MACD histogram thresholds for entry
    MACD_HIST_MIN = 0  # Histogram must be positive for long, negative for short
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # Minimum bars for all indicators to be valid
    first_valid = max(80, 55, 28, 28)  # ADX needs 2*period, EMA needs 55, MACD needs 26+9
    
    # Track entry prices for trailing stop
    entry_price_long = np.zeros(n)
    entry_price_short = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(macd_hist[i]) or np.isnan(adx_1h[i]) or np.isnan(atr_1h[i]) or np.isnan(trend_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        adx_val = adx_1h[i]
        atr = atr_1h[i]
        price = close[i]
        hist = macd_hist[i]
        prev_hist = macd_hist[i - 1] if i > 0 else 0
        
        # ADX filter - only trade when trend strength is real
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            continue
        
        # ATR filter - avoid extreme volatility
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            continue
        
        if trend == 1:  # 4h uptrend
            # Check for MACD histogram cross above zero (momentum confirmation)
            if hist > MACD_HIST_MIN and prev_hist <= MACD_HIST_MIN:
                # Fresh long entry
                signals[i] = SIZE_FULL
                entry_price_long[i] = price
                highest_since_entry[i] = price
            elif signals[i - 1] > 0:
                # Hold existing long position
                # Update highest price since entry
                entry_idx = max(0, i - 100)
                recent_entries = entry_price_long[entry_idx:i]
                valid_entries = recent_entries[recent_entries > 0]
                
                if len(valid_entries) > 0:
                    entry_price = valid_entries[-1]
                    
                    # Update highest price for trailing stop
                    if i > entry_idx:
                        highest_since_entry[i] = max(highest_since_entry[i - 1], price)
                    
                    # Trailing stop: 2.5*ATR from highest price
                    stoploss_price = highest_since_entry[i] - ATR_STOP_MULT * atr
                    
                    if price < stoploss_price:
                        signals[i] = 0.0  # Stoploss triggered
                    else:
                        # Check if MACD histogram turned negative (exit signal)
                        if hist < 0:
                            signals[i] = 0.0  # Momentum reversal
                        else:
                            signals[i] = signals[i - 1]  # Hold position
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
                
        elif trend == -1:  # 4h downtrend
            # Check for MACD histogram cross below zero (momentum confirmation)
            if hist < -MACD_HIST_MIN and prev_hist >= -MACD_HIST_MIN:
                # Fresh short entry
                signals[i] = -SIZE_FULL
                entry_price_short[i] = price
                lowest_since_entry[i] = price
            elif signals[i - 1] < 0:
                # Hold existing short position
                # Update lowest price since entry
                entry_idx = max(0, i - 100)
                recent_entries = entry_price_short[entry_idx:i]
                valid_entries = recent_entries[recent_entries > 0]
                
                if len(valid_entries) > 0:
                    entry_price = valid_entries[-1]
                    
                    # Update lowest price for trailing stop
                    if i > entry_idx:
                        lowest_since_entry[i] = min(lowest_since_entry[i - 1], price)
                    
                    # Trailing stop: 2.5*ATR from lowest price
                    stoploss_price = lowest_since_entry[i] + ATR_STOP_MULT * atr
                    
                    if price > stoploss_price:
                        signals[i] = 0.0  # Stoploss triggered
                    else:
                        # Check if MACD histogram turned positive (exit signal)
                        if hist > 0:
                            signals[i] = 0.0  # Momentum reversal
                        else:
                            signals[i] = signals[i - 1]  # Hold position
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
    
    return signals