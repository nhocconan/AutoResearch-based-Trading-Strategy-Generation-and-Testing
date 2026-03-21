#!/usr/bin/env python3
"""
EXPERIMENT #005 - MTF Daily Trend + 4h Supertrend + 1h RSI/MACD (1h primary)
==================================================================================================
Hypothesis: Use 1h as primary timeframe (fewer trades, less noise than 15m/30m) with:
- 1d SMA(50) for long-term trend direction (strongest filter)
- 4h Supertrend for medium-term trend confirmation
- 1h RSI pullback + MACD histogram for entry timing
- ATR-based stoploss (2*ATR) and take profit (2R reduce, trail at 1R)

Why this should work:
- 1h timeframe has proven success in prior experiments (#031, #034)
- Daily trend filter eliminates counter-trend trades during major reversals
- 4h Supertrend adds intermediate trend confirmation
- Fewer trades = less fee drag (0.10% per round trip)
- Discrete signal levels (0, ±0.25, ±0.35) minimize churn costs
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_daily_sma_supertrend_rsi_macd_1h_4h_1d_v1"
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
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().values
    
    rs = np.zeros(n)
    for i in range(n):
        if avg_loss[i] == 0:
            rs[i] = 100
        else:
            rs[i] = avg_gain[i] / avg_loss[i]
    
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = np.zeros(n)
    lower_band = np.zeros(n)
    
    for i in range(n):
        mid = (high[i] + low[i]) / 2
        upper_band[i] = mid + multiplier * atr[i]
        lower_band[i] = mid - multiplier * atr[i]
    
    supertrend[period - 1] = lower_band[period - 1]
    
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


def calculate_macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD line, signal line, and histogram"""
    n = len(close)
    if n < slow + signal:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    
    ema_fast = pd.Series(close).ewm(span=fast, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=slow, adjust=False).mean().values
    
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_sma(close, period=50):
    """Calculate Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    macd_1h, macd_signal_1h, macd_hist_1h = calculate_macd(close, fast=12, slow=26, signal=9)
    
    # Get 4h data using mtf_data helper
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h Supertrend for trend direction
        _, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
        
        # Align 4h indicators to 1h timeframe
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
    except Exception:
        st_direction_4h_aligned = np.zeros(n)
    
    # Get 1d data using mtf_data helper for long-term trend
    try:
        df_1d = get_htf_data(prices, '1d')
        c_1d = df_1d['close'].values
        
        # 1d SMA(50) for long-term trend
        sma_1d = calculate_sma(c_1d, period=50)
        
        # Align 1d SMA to 1h timeframe
        sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
        
        # Calculate 1d trend direction (price vs SMA)
        trend_1d = np.zeros(n)
        for i in range(n):
            if i < len(c_1d) and i < len(sma_1d_aligned):
                if c_1d[min(i, len(c_1d)-1)] > sma_1d_aligned[i]:
                    trend_1d[i] = 1
                elif c_1d[min(i, len(c_1d)-1)] < sma_1d_aligned[i]:
                    trend_1d[i] = -1
    except Exception:
        trend_1d = np.zeros(n)
        sma_1d_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 35
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 65
    
    # MACD histogram threshold for momentum confirmation
    MACD_HIST_MIN = 0.0
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    first_valid = max(200, 50 * 24, 26 + 9)  # Need 50 days of daily data
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned MTF values
        st_trend_4h = st_direction_4h_aligned[i] if i < len(st_direction_4h_aligned) else 0
        macd_hist_val = macd_hist_1h[i] if i < len(macd_hist_1h) else 0
        rsi_val = rsi_1h[i] if i < len(rsi_1h) else 50
        daily_trend = trend_1d[i] if i < len(trend_1d) else 0
        
        # 1d trend filter (must align with trade direction)
        # Only long if daily trend is bullish, only short if daily trend is bearish
        if daily_trend == 0:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # 4h Supertrend must align with daily trend
        if st_trend_4h != 0 and st_trend_4h != daily_trend:
            signals[i] = 0.0
            position_side[i] = 0
            continue
        
        # Check stoploss and take profit for existing positions
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
            price = close[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            else:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (2.0*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_1h[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * atr_1h[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_1h[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit check (2R) - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_1h[i]
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    continue
                
                # Trail stop at 1R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * atr_1h[i]
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
        
        # Entry logic: Daily trend + 4h Supertrend + 1h RSI/MACD
        price = close[i]
        
        if daily_trend == 1 and st_trend_4h == 1:  # Bullish trend
            # MACD histogram positive on 1h (momentum confirmation)
            # RSI pullback on 1h (not overbought)
            if (macd_hist_val > MACD_HIST_MIN and 
                RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX):
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        elif daily_trend == -1 and st_trend_4h == -1:  # Bearish trend
            # MACD histogram negative on 1h (momentum confirmation)
            # RSI pullback on 1h (not oversold)
            if (macd_hist_val < -MACD_HIST_MIN and 
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