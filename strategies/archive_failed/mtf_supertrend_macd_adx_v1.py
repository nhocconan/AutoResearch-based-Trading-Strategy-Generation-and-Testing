#!/usr/bin/env python3
"""
EXPERIMENT #018 - Supertrend + MACD + ADX Filter + ATR Stop
===============================================================================
Hypothesis: Supertrend provides clearer trend signals than Donchian in strong trending markets.
MACD histogram crosses give earlier entry signals than RSI pullbacks.
ADX filter avoids trading in choppy/ranging markets (ADX < 25).
This combination should reduce whipsaws and improve win rate.

Key differences from #017:
- Supertrend(ATR=10, mult=3) instead of Donchian(20) for trend
- MACD histogram cross instead of RSI pullback for entries
- ADX(14) > 25 filter to avoid choppy markets
- Cleaner stoploss logic with proper entry price tracking
- Same discrete position sizing (0.0, ±0.25, ±0.35)

Why this might beat Sharpe=5.4:
- Supertrend is more robust in strong trends (less whipsaw than Donchian)
- MACD histogram gives momentum confirmation earlier than RSI
- ADX filter removes low-quality trades in ranging markets
- Multi-timeframe logic proven in previous experiments
"""

import numpy as np
import pandas as pd

name = "mtf_supertrend_macd_adx_v1"
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


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator for trend direction"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    hl2 = (high + low) / 2
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    supertrend = np.zeros(n)
    trend = np.zeros(n)  # 1 = bullish, -1 = bearish
    
    for i in range(period, n):
        upper_band[i] = hl2[i] + multiplier * atr[i]
        lower_band[i] = hl2[i] - multiplier * atr[i]
        
        if i == period:
            supertrend[i] = upper_band[i]
            trend[i] = -1
        else:
            if close[i - 1] <= supertrend[i - 1]:
                supertrend[i] = min(upper_band[i], supertrend[i - 1])
                if close[i] > supertrend[i]:
                    supertrend[i] = lower_band[i]
                    trend[i] = 1
                else:
                    trend[i] = -1
            else:
                supertrend[i] = max(lower_band[i], supertrend[i - 1])
                if close[i] < supertrend[i]:
                    supertrend[i] = upper_band[i]
                    trend[i] = -1
                else:
                    trend[i] = 1
    
    return supertrend, upper_band, lower_band, trend


def calculate_macd(close, fast=12, slow=26, signal_period=9):
    """Calculate MACD histogram"""
    n = len(close)
    if n < slow + signal_period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal_period, min_periods=signal_period).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_adx(high, low, close, period=14):
    """Calculate ADX for trend strength"""
    n = len(close)
    if n < period * 2:
        return np.zeros(n)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    plus_smooth = pd.Series(plus_dm).ewm(span=period, min_periods=period).mean().values
    minus_smooth = pd.Series(minus_dm).ewm(span=period, min_periods=period).mean().values
    tr_smooth = pd.Series(tr).ewm(span=period, min_periods=period).mean().values
    
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = tr_smooth > 0
    plus_di[mask] = 100 * plus_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100 * minus_smooth[mask] / tr_smooth[mask]
    
    dx = np.zeros(n)
    mask2 = (plus_di + minus_di) > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / (plus_di[mask2] + minus_di[mask2])
    
    adx = pd.Series(dx).ewm(span=period, min_periods=period).mean().values
    
    return adx


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal_period=9)
    adx_1h = calculate_adx(high, low, close, period=14)
    
    # 4h Supertrend for trend filter (resample 1h → 4h)
    df_1h = pd.DataFrame({
        'open': close,
        'high': high,
        'low': low,
        'close': close
    })
    df_1h.index = pd.date_range(start='2021-01-01', periods=n, freq='1h')
    
    df_4h = df_1h.resample('4h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).dropna()
    
    h_4h = df_4h['high'].values
    l_4h = df_4h['low'].values
    c_4h = df_4h['close'].values
    n_4h = len(c_4h)
    
    # Calculate 4h Supertrend
    if n_4h > 10:
        _, _, _, trend_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
    else:
        trend_4h = np.zeros(n_4h)
    
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
    SIZE_FULL = 0.35
    SIZE_HALF = 0.25
    
    # MACD thresholds for entry
    MACD_HIST_THRESHOLD = 0  # Histogram must cross above/below zero
    
    # ADX threshold for trend strength
    ADX_MIN = 25  # Only trade when ADX > 25 (strong trend)
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(100, 35, 28)  # Wait for all indicators
    
    # Track entry prices and position for trailing stop
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    prev_signal = 0.0
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(macd_hist[i]) or np.isnan(adx_1h[i]):
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            prev_signal = signals[i]
            continue
        
        trend = trend_1h[i]
        adx_val = adx_1h[i]
        macd_histogram = macd_hist[i]
        macd_histogram_prev = macd_hist[i - 1] if i > 0 else 0
        atr = atr_1h[i]
        price = close[i]
        
        # ADX filter - only trade in strong trends
        if adx_val < ADX_MIN:
            # If we have a position, check stoploss; otherwise stay flat
            if position_side[i - 1] != 0:
                prev_side = position_side[i - 1]
                prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
                
                if prev_side == 1:
                    stoploss_price = prev_entry - ATR_STOP_MULT * atr
                    if price < stoploss_price:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                    else:
                        signals[i] = prev_signal
                        position_side[i] = prev_side
                        entry_price[i] = prev_entry
                elif prev_side == -1:
                    stoploss_price = prev_entry + ATR_STOP_MULT * atr
                    if price > stoploss_price:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                    else:
                        signals[i] = prev_signal
                        position_side[i] = prev_side
                        entry_price[i] = prev_entry
            else:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
            
            prev_signal = signals[i]
            continue
        
        # Check trailing stop for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            
            if prev_side == 1:  # Long position
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    prev_signal = 0.0
                    continue
            elif prev_side == -1:  # Short position
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    prev_signal = 0.0
                    continue
        
        # MACD histogram cross for entry timing
        macd_bullish_cross = (macd_histogram > 0) and (macd_histogram_prev <= 0)
        macd_bearish_cross = (macd_histogram < 0) and (macd_histogram_prev >= 0)
        
        if trend == 1:  # 4h uptrend
            if macd_bullish_cross:
                # Fresh bullish cross - full position
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
            elif macd_histogram > 0:
                # MACD positive but no fresh cross - hold or half position
                if position_side[i - 1] == 1:
                    signals[i] = prev_signal
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = price
            else:
                # MACD negative - exit or hold existing
                if position_side[i - 1] == 1:
                    signals[i] = prev_signal
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    
        elif trend == -1:  # 4h downtrend
            if macd_bearish_cross:
                # Fresh bearish cross - full short
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
            elif macd_histogram < 0:
                # MACD negative but no fresh cross - hold or half short
                if position_side[i - 1] == -1:
                    signals[i] = prev_signal
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = price
            else:
                # MACD positive - exit or hold existing
                if position_side[i - 1] == -1:
                    signals[i] = prev_signal
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
        
        prev_signal = signals[i]
    
    return signals