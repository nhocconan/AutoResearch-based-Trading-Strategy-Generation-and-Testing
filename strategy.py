#!/usr/bin/env python3
"""
EXPERIMENT #018 - KAMA Adaptive Trend + MACD Histogram + ADX Strength Filter
===============================================================================
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market volatility,
becoming fast in trending markets and slow in choppy ones. Combined with MACD
histogram crosses for entry timing and ADX for trend strength, this should
reduce whipsaw while capturing strong trends.

Key innovations vs mtf_supertrend_macd_adx_v1:
- KAMA instead of Supertrend for adaptive trend following (less lag in trends)
- MACD histogram cross (not just signal line cross) for earlier entry signals
- ADX > 25 filter ensures we only trade when trend has genuine strength
- Proper ATR trailing stop with signal→0 when price moves 2.5*ATR against
- Discrete position sizing (0.0, ±0.25, ±0.35) to minimize churn costs

Why this might beat Sharpe=1.278:
- KAMA's adaptive nature reduces false signals in choppy markets
- MACD histogram leads signal line crosses (earlier entries)
- ADX filter avoids trading in weak/ranging markets
- Multi-timeframe: 4h KAMA trend + 1h MACD entries (proven approach)
- Conservative position sizing (max 0.35) controls drawdown
"""

import numpy as np
import pandas as pd

name = "mtf_kama_macd_adx_atr_v1"
timeframe = "1h"
leverage = 1.0


def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average
    Adapts to market volatility - fast in trends, slow in chop
    """
    n = len(close)
    kama = np.zeros(n)
    
    if n < period + slow_period:
        return kama
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if volatility > 0:
            er[i] = price_change / volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)
    
    # KAMA calculation
    kama[period] = close[period]
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    
    ema_fast = pd.Series(close).ewm(span=fast, min_periods=fast).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, min_periods=slow).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, min_periods=signal).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index) for trend strength"""
    n = len(close)
    adx = np.zeros(n)
    
    if n < period * 2:
        return adx
    
    # Calculate True Range and Directional Movement
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    
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
    
    # Smooth TR, +DM, -DM using Wilder's method
    atr = np.zeros(n)
    plus_dm_smooth = np.zeros(n)
    minus_dm_smooth = np.zeros(n)
    
    atr[period - 1] = np.mean(tr[1:period])
    plus_dm_smooth[period - 1] = np.mean(plus_dm[1:period])
    minus_dm_smooth[period - 1] = np.mean(minus_dm[1:period])
    
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        plus_dm_smooth[i] = (plus_dm_smooth[i - 1] * (period - 1) + plus_dm[i]) / period
        minus_dm_smooth[i] = (minus_dm_smooth[i - 1] * (period - 1) + minus_dm[i]) / period
    
    # Calculate +DI, -DI, and DX
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    dx = np.zeros(n)
    
    mask = atr > 0
    plus_di[mask] = 100 * plus_dm_smooth[mask] / atr[mask]
    minus_di[mask] = 100 * minus_dm_smooth[mask] / atr[mask]
    
    di_sum = plus_di + minus_di
    mask2 = di_sum > 0
    dx[mask2] = 100 * np.abs(plus_di[mask2] - minus_di[mask2]) / di_sum[mask2]
    
    # Calculate ADX (smoothed DX)
    adx[period * 2 - 1] = np.mean(dx[period:period * 2])
    for i in range(period * 2, n):
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
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    macd_line, macd_signal, macd_hist = calculate_macd(close, fast=12, slow=26, signal=9)
    adx_1h = calculate_adx(high, low, close, period=14)
    
    # 4h KAMA for adaptive trend (resample 1h → 4h)
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
    
    # Calculate 4h KAMA for adaptive trend
    kama_4h = calculate_kama(c_4h, period=10, fast_period=2, slow_period=30)
    
    # Determine 4h trend direction from KAMA slope
    trend_4h = np.zeros(len(c_4h))
    for i in range(15, len(c_4h)):
        if kama_4h[i] > kama_4h[i - 1]:
            trend_4h[i] = 1  # Bullish (KAMA sloping up)
        elif kama_4h[i] < kama_4h[i - 1]:
            trend_4h[i] = -1  # Bearish (KAMA sloping down)
    
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
    SIZE_FULL = 0.35   # Full position when all conditions met
    SIZE_HALF = 0.25   # Reduced position in marginal conditions
    
    # ADX threshold for trend strength
    ADX_MIN = 25       # Only trade when ADX > 25 (strong trend)
    
    # MACD histogram thresholds for entry
    MACD_HIST_THRESHOLD = 0  # Cross above/below zero
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(80, 30, 28)  # Wait for all indicators
    
    # Track entry prices for trailing stop logic
    entry_price = np.zeros(n)
    position_side = np.zeros(n)  # 1 for long, -1 for short, 0 for flat
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(macd_hist[i]) or np.isnan(adx_1h[i]):
            signals[i] = 0.0
            continue
        
        trend = trend_1h[i]
        adx_val = adx_1h[i]
        atr = atr_1h[i]
        price = close[i]
        macd_histogram = macd_hist[i]
        
        # Check if we have previous MACD histogram value for cross detection
        if i > 0:
            prev_macd_hist = macd_hist[i - 1]
        else:
            prev_macd_hist = 0
        
        # ATR filter - avoid trading when ATR is extremely high
        if atr > 0 and atr / price > 0.05:  # ATR > 5% of price = too volatile
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
            continue
        
        # Check trailing stop for existing positions
        if i > 0 and position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else price
            
            if prev_side == 1:  # Long position
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    continue
            elif prev_side == -1:  # Short position
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    continue
        
        # ADX filter - only trade when trend has strength
        if adx_val < ADX_MIN:
            # If we have a position, hold it; otherwise stay flat
            if i > 0 and position_side[i - 1] != 0:
                signals[i] = signals[i - 1]
                position_side[i] = position_side[i - 1]
                entry_price[i] = entry_price[i - 1]
            else:
                signals[i] = 0.0
                position_side[i] = 0
            continue
        
        if trend == 1:  # 4h uptrend (KAMA sloping up)
            # MACD histogram cross above zero for long entry
            macd_bullish_cross = (prev_macd_hist <= 0 and macd_histogram > 0)
            macd_bullish = macd_histogram > 0
            
            if macd_bullish_cross:
                # Fresh bullish cross - full position
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
            elif macd_bullish:
                # MACD still positive - hold or enter half
                if i > 0 and position_side[i - 1] == 1:
                    signals[i] = signals[i - 1]
                    position_side[i] = 1
                    entry_price[i] = entry_price[i - 1]
                elif i > 0 and position_side[i - 1] == 0:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = price
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
            else:
                # MACD turned negative - exit long
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                    
        elif trend == -1:  # 4h downtrend (KAMA sloping down)
            # MACD histogram cross below zero for short entry
            macd_bearish_cross = (prev_macd_hist >= 0 and macd_histogram < 0)
            macd_bearish = macd_histogram < 0
            
            if macd_bearish_cross:
                # Fresh bearish cross - full short
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
            elif macd_bearish:
                # MACD still negative - hold or enter half short
                if i > 0 and position_side[i - 1] == -1:
                    signals[i] = signals[i - 1]
                    position_side[i] = -1
                    entry_price[i] = entry_price[i - 1]
                elif i > 0 and position_side[i - 1] == 0:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = price
                else:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
            else:
                # MACD turned positive - exit short
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
        else:  # No clear trend
            signals[i] = 0.0
            position_side[i] = 0
            entry_price[i] = 0
    
    return signals