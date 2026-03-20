#!/usr/bin/env python3
"""
EXPERIMENT #003 - Multi-Timeframe Donchian Breakout + MACD Entry + ADX Filter
=============================================================================
Hypothesis: Donchian Channel(20) on 4h provides cleaner trend signals than HMA/Supertrend
by capturing actual breakouts. MACD histogram cross on 1h gives precise entry timing.
ADX(14) > 25 ensures we only trade when trend has sufficient strength.

Key differences from mtf_hma_rsi_zscore_v1:
- Donchian(20) breakout instead of HMA trend (captures momentum breakouts better)
- MACD histogram cross instead of RSI pullback (momentum-based entries)
- ADX strength filter to avoid weak/choppy trends
- ATR trailing stoploss for risk management

Why this might beat Sharpe=1.768:
- Donchian breakouts catch strong trends early
- MACD histogram is more responsive than RSI for momentum
- ADX filter reduces whipsaw in ranging markets
- ATR stoploss limits drawdown on failed breakouts
"""

import numpy as np
import pandas as pd

name = "mtf_donchian_macd_adx_v1"
timeframe = "1h"
leverage = 1.0


def calculate_donchian(high, low, period=20):
    """
    Calculate Donchian Channel
    Upper = highest high over period
    Lower = lowest low over period
    Middle = (Upper + Lower) / 2
    """
    n = len(high)
    upper = np.zeros(n)
    lower = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
    
    middle = (upper + lower) / 2
    return upper, lower, middle


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    
    # EMA helper
    def ema(series, period):
        result = np.zeros(n)
        multiplier = 2 / (period + 1)
        result[0] = series[0]
        for i in range(1, n):
            result[i] = (series[i] - result[i - 1]) * multiplier + result[i - 1]
        return result
    
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    
    macd_line = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    signal_line = ema(macd_line, signal)
    
    # Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength"""
    n = len(close)
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        # True Range
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
        
        # Directional Movement
        if high[i] - high[i - 1] > low[i - 1] - low[i]:
            plus_dm[i] = max(high[i] - high[i - 1], 0)
        else:
            plus_dm[i] = 0
            
        if low[i - 1] - low[i] > high[i] - high[i - 1]:
            minus_dm[i] = max(low[i - 1] - low[i], 0)
        else:
            minus_dm[i] = 0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (RMA)
    def rma(series, period):
        result = np.zeros(n)
        result[period - 1] = np.sum(series[:period]) / period
        for i in range(period, n):
            result[i] = (result[i - 1] * (period - 1) + series[i]) / period
        return result
    
    tr_smooth = rma(tr, period)
    plus_dm_smooth = rma(plus_dm, period)
    minus_dm_smooth = rma(minus_dm, period)
    
    # Directional Indicators
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    mask = tr_smooth > 0
    plus_di[mask] = 100 * plus_dm_smooth[mask] / tr_smooth[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / tr_smooth[mask]
    
    # DX and ADX
    dx = np.zeros(n)
    mask2 = (plus_di + minus_di) > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / (plus_di[mask2] + minus_di[mask2])
    
    adx = rma(dx, period)
    
    return adx, plus_di, minus_di


def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    n = len(close)
    tr = np.zeros(n)
    
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    # RMA smoothing
    atr = np.zeros(n)
    atr[period - 1] = np.sum(tr[:period]) / period
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    macd_line, signal_line, histogram = calculate_macd(close, fast=12, slow=26, signal=9)
    adx, plus_di, minus_di = calculate_adx(high, low, close, period=14)
    atr_1h = calculate_atr(high, low, close, period=14)
    
    # 4h Donchian for trend filter (resample 1h → 4h)
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
    
    n_4h = len(df_4h)
    
    # Calculate 4h Donchian(20)
    upper_4h, lower_4h, middle_4h = calculate_donchian(
        df_4h['high'].values,
        df_4h['low'].values,
        period=20
    )
    
    # Calculate 4h trend direction
    trend_4h = np.zeros(n_4h)
    for i in range(19, n_4h):
        if upper_4h[i] > 0:
            if df_4h['close'].values[i] > middle_4h[i]:
                trend_4h[i] = 1  # Bullish
            elif df_4h['close'].values[i] < middle_4h[i]:
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
    SIZE_HALF = 0.20   # Reduced position in marginal conditions
    
    # ADX threshold for trend strength
    ADX_MIN = 25       # Only trade when ADX > 25 (strong trend)
    
    # MACD histogram thresholds for entry
    MACD_LONG_THRESHOLD = 0    # Histogram crosses above 0
    MACD_SHORT_THRESHOLD = 0   # Histogram crosses below 0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5  # Stoploss at 2.5 * ATR
    
    # Track entry prices for stoploss
    entry_price = np.zeros(n)
    position_high = np.zeros(n)  # Track highest price since entry (for longs)
    position_low = np.zeros(n)   # Track lowest price since entry (for shorts)
    
    first_valid = max(48, 26, 14)  # Wait for all indicators
    
    for i in range(first_valid, n):
        # Check for valid data
        if np.isnan(histogram[i]) or np.isnan(adx[i]) or np.isnan(trend_1h[i]) or np.isnan(atr_1h[i]):
            signals[i] = 0.0
            continue
        
        if i == 0 or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        adx_val = adx[i]
        hist_val = histogram[i]
        hist_prev = histogram[i - 1] if i > 0 else 0
        atr_val = atr_1h[i]
        current_price = close[i]
        
        # ADX filter - only trade strong trends
        if adx_val < ADX_MIN:
            signals[i] = 0.0
            entry_price[i] = 0.0
            continue
        
        # Get previous signal for stoploss logic
        prev_signal = signals[i - 1] if i > 0 else 0.0
        prev_entry = entry_price[i - 1] if i > 0 else 0.0
        
        # Initialize tracking if new position
        if prev_signal == 0 and signals[i] != 0:
            entry_price[i] = current_price
            position_high[i] = current_price
            position_low[i] = current_price
        elif prev_signal != 0:
            entry_price[i] = prev_entry
            position_high[i] = max(position_high[i - 1], current_price)
            position_low[i] = min(position_low[i - 1], current_price)
        
        # ATR Trailing Stoploss Logic
        if prev_signal > 0:  # Long position
            # Trail stop: highest price since entry - ATR_STOP_MULT * ATR
            trail_stop = position_high[i - 1] - ATR_STOP_MULT * atr_val
            if current_price < trail_stop:
                signals[i] = 0.0
                entry_price[i] = 0.0
                position_high[i] = 0.0
                position_low[i] = 0.0
                continue
        elif prev_signal < 0:  # Short position
            # Trail stop: lowest price since entry + ATR_STOP_MULT * ATR
            trail_stop = position_low[i - 1] + ATR_STOP_MULT * atr_val
            if current_price > trail_stop:
                signals[i] = 0.0
                entry_price[i] = 0.0
                position_high[i] = 0.0
                position_low[i] = 0.0
                continue
        
        # Entry logic based on trend and MACD
        if trend == 1:  # 4h uptrend
            # MACD histogram crossing above 0 or already positive
            if hist_val > MACD_LONG_THRESHOLD and (hist_prev <= 0 or prev_signal > 0):
                # Strong trend (ADX > 35) - full position
                if adx_val > 35:
                    signals[i] = SIZE_FULL
                else:
                    signals[i] = SIZE_HALF
            else:
                # Hold position if already in, otherwise flat
                if prev_signal > 0:
                    signals[i] = prev_signal
                else:
                    signals[i] = 0.0
        elif trend == -1:  # 4h downtrend
            # MACD histogram crossing below 0 or already negative
            if hist_val < MACD_SHORT_THRESHOLD and (hist_prev >= 0 or prev_signal < 0):
                # Strong trend (ADX > 35) - full short
                if adx_val > 35:
                    signals[i] = -SIZE_FULL
                else:
                    signals[i] = -SIZE_HALF
            else:
                # Hold position if already in, otherwise flat
                if prev_signal < 0:
                    signals[i] = prev_signal
                else:
                    signals[i] = 0.0
        else:  # No clear trend
            signals[i] = 0.0
            entry_price[i] = 0.0
    
    return signals