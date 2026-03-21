#!/usr/bin/env python3
"""
EXPERIMENT #020 - MTF Donchian+KAMA+RSI (15m+1h+4h Clean v1)
==================================================================================================
Hypothesis: Previous EMA crossover (#019) crashed due to complex position tracking arrays.
Let's try Donchian Channel for trend (breakout-based, proven in crypto) + KAMA for adaptive
momentum (better than MACD in ranging markets) + RSI pullback entries.

Key changes from #019:
- Trend: 4H Donchian Channel (20-period) instead of EMA crossover (cleaner breakout signals)
- Momentum: 1H KAMA (Kaufman Adaptive MA) instead of MACD (adapts to volatility)
- Entry: 15m RSI pullback (same proven logic)
- Simplified position tracking (no complex state arrays that caused crash)
- Stoploss: 2.5*ATR trailing stop (slightly wider than #019's 2.0*ATR)
- Position size: 0.30 max (slightly more conservative)

Why this should work:
- Donchian channels excel in crypto trending markets (breakout capture)
- KAMA reduces whipsaws in sideways markets better than EMA/MACD
- Simpler signal logic = fewer indexing errors
- Based on #009 (Sharpe=0.029) which used Supertrend+KAMA successfully
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf


def calculate_atr(high, low, close, period=14):
    """Calculate ATR using Wilder's smoothing"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1])
        )
    
    atr = np.zeros(n)
    atr[period] = np.mean(tr[1:period + 1])
    
    for i in range(period + 1, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    
    return atr


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < period + slow:
        return np.zeros(n)
    
    kama = np.zeros(n)
    
    # Calculate Efficiency Ratio
    er = np.zeros(n)
    for i in range(period, n):
        price_change = abs(close[i] - close[i - period])
        sum_volatility = np.sum(np.abs(np.diff(close[max(0, i - period):i + 1])))
        
        if sum_volatility > 0:
            er[i] = price_change / sum_volatility
        else:
            er[i] = 0
    
    # Calculate Smoothing Constant
    sc = np.zeros(n)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    
    for i in range(period, n):
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_rsi(close, period=14):
    """Calculate RSI"""
    n = len(close)
    if n < period + 1:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    
    avg_gain[period] = np.mean(gain[:period + 1])
    avg_loss[period] = np.mean(loss[:period + 1])
    
    for i in range(period + 1, n):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gain[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + loss[i]) / period
    
    rs = np.zeros(n)
    for i in range(period, n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_donchian(high, low, period=20):
    """Calculate Donchian Channel (upper, lower, middle)"""
    n = len(high)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    upper = np.zeros(n)
    lower = np.zeros(n)
    middle = np.zeros(n)
    
    for i in range(period - 1, n):
        upper[i] = np.max(high[i - period + 1:i + 1])
        lower[i] = np.min(low[i - period + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    return upper, lower, middle


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Initialize signals array
    signals = np.zeros(n)
    
    # 15m indicators for entry timing
    atr_15m = calculate_atr(high, low, close, period=14)
    rsi_15m = calculate_rsi(close, period=14)
    
    # Get 1h data using mtf_data helper
    kama_1h_aligned = np.zeros(n)
    try:
        df_1h = get_htf_data(prices, '1h')
        if df_1h is not None and len(df_1h) > 0:
            close_1h = df_1h['close'].values
            kama_1h = calculate_kama(close_1h, period=10, fast=2, slow=30)
            kama_1h_aligned = align_htf_to_ltf(prices, df_1h, kama_1h)
    except Exception:
        kama_1h_aligned = np.zeros(n)
    
    # Get 4h data using mtf_data helper
    donchian_trend_aligned = np.zeros(n)
    try:
        df_4h = get_htf_data(prices, '4h')
        if df_4h is not None and len(df_4h) > 0:
            high_4h = df_4h['high'].values
            low_4h = df_4h['low'].values
            close_4h = df_4h['close'].values
            
            upper_4h, lower_4h, middle_4h = calculate_donchian(high_4h, low_4h, period=20)
            
            # Determine trend direction from Donchian position
            trend_4h = np.zeros(len(close_4h))
            for i in range(20, len(close_4h)):
                if close_4h[i] > middle_4h[i] and close_4h[i] > close_4h[i - 1]:
                    trend_4h[i] = 1  # Bullish
                elif close_4h[i] < middle_4h[i] and close_4h[i] < close_4h[i - 1]:
                    trend_4h[i] = -1  # Bearish
                else:
                    trend_4h[i] = trend_4h[i - 1] if i > 0 else 0
            
            donchian_trend_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    except Exception:
        donchian_trend_aligned = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # Minimum bars for valid signals
    first_valid = max(100, 20 * 4, 14 * 2)
    
    # Track position state (simplified to avoid indexing errors)
    current_position = 0  # 0=none, 1=long, -1=short
    entry_price = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_15m[i]) or np.isnan(rsi_15m[i]) or atr_15m[i] <= 0:
            signals[i] = 0.0
            current_position = 0
            entry_price = 0.0
            tp_triggered = False
            continue
        
        trend = donchian_trend_aligned[i]
        rsi_val = rsi_15m[i]
        atr = atr_15m[i]
        price = close[i]
        kama_1h = kama_1h_aligned[i]
        
        # Check stoploss and take profit for existing positions
        if current_position != 0:
            # Update highest/lowest since entry
            if current_position == 1:
                highest_since_entry = max(highest_since_entry, price)
                lowest_since_entry = min(lowest_since_entry, price) if lowest_since_entry > 0 else price
                
                # Stoploss check
                stoploss_price = entry_price - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    current_position = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = entry_price + 2 * ATR_STOP_MULT * atr
                if not tp_triggered and price >= tp_price:
                    signals[i] = SIZE_HALF
                    tp_triggered = True
                    continue
                
                # Trail stop at 1R profit
                if tp_triggered:
                    trail_stop = highest_since_entry - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        current_position = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
                
                # Hold position
                signals[i] = SIZE_FULL if not tp_triggered else SIZE_HALF
                
            elif current_position == -1:
                highest_since_entry = max(highest_since_entry, price) if highest_since_entry > 0 else price
                lowest_since_entry = min(lowest_since_entry, price)
                
                # Stoploss check
                stoploss_price = entry_price + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    current_position = 0
                    entry_price = 0.0
                    tp_triggered = False
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = entry_price - 2 * ATR_STOP_MULT * atr
                if not tp_triggered and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    tp_triggered = True
                    continue
                
                # Trail stop at 1R profit
                if tp_triggered:
                    trail_stop = lowest_since_entry + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        current_position = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
                
                # Hold position
                signals[i] = -SIZE_FULL if not tp_triggered else -SIZE_HALF
            
            continue
        
        # Entry logic: 4H Donchian trend + 1H KAMA momentum + 15m RSI pullback
        if trend == 1:  # Bullish trend on 4H
            if (price > kama_1h and  # 1H KAMA momentum positive
                RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):  # 15m RSI pullback
                signals[i] = SIZE_FULL
                current_position = 1
                entry_price = price
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
                
        elif trend == -1:  # Bearish trend on 4H
            if (price < kama_1h and  # 1H KAMA momentum negative
                RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):  # 15m RSI pullback
                signals[i] = -SIZE_FULL
                current_position = -1
                entry_price = price
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
        
        else:
            signals[i] = 0.0
            current_position = 0
    
    return signals


name = "mtf_donchian_kama_rsi_15m_1h_4h_v1"
timeframe = "15m"
leverage = 1.0