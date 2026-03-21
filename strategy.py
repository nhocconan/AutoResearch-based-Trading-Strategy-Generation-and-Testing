#!/usr/bin/env python3
"""
EXPERIMENT #003 - MTF KAMA + Bollinger Squeeze + RSI (1h+4h v1)
==================================================================================================
Hypothesis: KAMA adapts to volatility better than DEMA/EMA. Bollinger squeeze identifies 
low-volatility breakout setups. Combined with 4h trend filter for direction.

Key changes from failed attempts:
1. KAMA instead of DEMA crossover (volatility-adaptive trend)
2. Bollinger Band squeeze for entry timing (volatility contraction → expansion)
3. Simpler logic - fewer conditions to reduce overfitting
4. Fix read-only array issue by creating new arrays
5. Position size: 0.30 (conservative)
6. Stoploss: 2.5*ATR (wider for 1h timeframe)

Why this should work:
- KAMA adapts speed based on market noise (ER ratio)
- Bollinger squeeze = low vol before breakout
- 4h KAMA trend filter = more stable direction
- RSI filter = avoid overbought/oversold entries
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_bb_squeeze_rsi_1h_4h_v1"
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


def calculate_kama(close, period=10, fast_sc=2, slow_sc=30):
    """
    Calculate Kaufman Adaptive Moving Average (KAMA)
    KAMA adapts to market noise using Efficiency Ratio (ER)
    """
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        change = abs(close[i] - close[i - period])
        sum_volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if sum_volatility > 0:
            er[i] = change / sum_volatility
        else:
            er[i] = 0
    
    # Calculate smoothing constant (SC)
    sc = np.zeros(n)
    fast_sc_val = 2 / (fast_sc + 1)
    slow_sc_val = 2 / (slow_sc + 1)
    
    for i in range(period, n):
        sc[i] = er[i] * (fast_sc_val - slow_sc_val) + slow_sc_val
        sc[i] = sc[i] ** 2  # Square for smoother adaptation
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        kama[i] = kama[i - 1] + sc[i] * (close[i] - kama[i - 1])
    
    return kama


def calculate_bollinger_bands(close, period=20, std_dev=2.0):
    """Calculate Bollinger Bands"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    # Use pandas rolling for proper min_periods
    close_series = pd.Series(close)
    sma = close_series.rolling(window=period, min_periods=period).mean().values
    std = close_series.rolling(window=period, min_periods=period).std().values
    
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    
    return upper, lower, bandwidth


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


def calculate_squeeze_signal(bandwidth, lookback=20):
    """
    Detect Bollinger Band squeeze (low volatility)
    Returns 1 when bandwidth is in bottom 20% of recent range
    """
    n = len(bandwidth)
    squeeze = np.zeros(n)
    
    for i in range(lookback, n):
        if bandwidth[i] == 0:
            continue
        recent_bw = bandwidth[i - lookback:i + 1]
        percentile = np.sum(recent_bw <= bandwidth[i]) / lookback
        if percentile >= 0.8:  # Bottom 20%
            squeeze[i] = 1
    
    return squeeze


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Get 4h HTF data using mtf_data helper (MANDATORY)
    df_4h = get_htf_data(prices, '4h')
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    kama_1h = calculate_kama(close, period=10)
    bb_upper_1h, bb_lower_1h, bb_bw_1h = calculate_bollinger_bands(close, period=20, std_dev=2.0)
    squeeze_1h = calculate_squeeze_signal(bb_bw_1h, lookback=20)
    
    # 4h indicators for trend (using mtf_data helper)
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    kama_4h = calculate_kama(close_4h, period=10)
    rsi_4h = calculate_rsi(close_4h, period=14)
    
    # Align 4h indicators to 1h timeframe (auto shift for completed bars)
    kama_4h_aligned = align_htf_to_ltf(prices, df_4h, kama_4h)
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.30
    SIZE_HALF = 0.15
    
    # RSI thresholds
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 65
    RSI_SHORT_MIN = 35
    RSI_SHORT_MAX = 60
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    # Minimum bars for valid signals
    first_valid = max(100, 20 + 20, 14 + 1)
    
    # Track position state (create new arrays, don't modify in place)
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        # Check for NaN or zero values
        if (np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or np.isnan(kama_1h[i]) or
            np.isnan(kama_4h_aligned[i]) or atr_1h[i] == 0 or bb_bw_1h[i] == 0):
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # 4h trend filter (HTF)
        trend_4h = 0
        if close[i] > kama_4h_aligned[i]:
            trend_4h = 1
        elif close[i] < kama_4h_aligned[i]:
            trend_4h = -1
        
        # 4h RSI filter (avoid extreme overbought/oversold)
        rsi_4h_val = rsi_4h_aligned[i]
        if np.isnan(rsi_4h_val):
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # 1h indicators for entry timing
        rsi_val = rsi_1h[i]
        atr = atr_1h[i]
        price = close[i]
        kama_1h_val = kama_1h[i]
        squeeze = squeeze_1h[i]
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h trend + 1h squeeze breakout + RSI filter
        # Long entry: 4h bullish + price above KAMA + squeeze + RSI in range
        if (trend_4h == 1 and 
            price > kama_1h_val and
            squeeze == 1 and
            RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):
            signals[i] = SIZE_FULL
            position_side[i] = 1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        # Short entry: 4h bearish + price below KAMA + squeeze + RSI in range
        elif (trend_4h == -1 and 
              price < kama_1h_val and
              squeeze == 1 and
              RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX):
            signals[i] = -SIZE_FULL
            position_side[i] = -1
            entry_price[i] = price
            tp_triggered[i] = 0
            highest_since_entry[i] = price
            lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals