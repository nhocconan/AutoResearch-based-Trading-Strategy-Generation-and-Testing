#!/usr/bin/env python3
"""
EXPERIMENT #029 - MTF KAMA+RSI+Supertrend (1h+4h Simplified v2)
==================================================================================================
Hypothesis: Previous HMA implementation had indexing issues. Switching to KAMA (Kaufman Adaptive MA)
which is more robust and adapts to market volatility. Simplified position tracking to avoid crashes.

Key changes from #028:
- Replace HMA with KAMA (more stable, adapts to volatility)
- Simplify position state tracking (avoid complex array indexing)
- Use 1h entries + 4h trend (proven combination)
- Position size: 0.25 (conservative for drawdown control)
- Add Z-score filter to avoid entering at extremes
- Cleaner stoploss/TP logic with proper array initialization

Why this should work:
- KAMA adapts to market regime (better than fixed HMA)
- Simpler position tracking reduces crash risk
- Z-score filter prevents chasing extended moves
- Conservative sizing (0.25) controls drawdown better than 0.30
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_kama_rsi_supertrend_zscore_1h_4h_v2"
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


def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    kama = np.zeros(n)
    kama[period - 1] = close[period - 1]
    
    for i in range(period, n):
        # Calculate efficiency ratio
        change = abs(close[i] - close[i - period])
        volatility = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        
        if volatility == 0:
            er = 1.0
        else:
            er = change / volatility
        
        # Calculate smoothing constant
        sc = (er * (2.0 / (fast + 1) - 2.0 / (slow + 1)) + 2.0 / (slow + 1)) ** 2
        
        kama[i] = kama[i - 1] + sc * (close[i] - kama[i - 1])
    
    return kama


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
        else:
            rsi[i] = 100 - (100 / (1 + avg_gain[i] / avg_loss[i]))
    
    return np.nan_to_num(rsi, nan=50.0)


def calculate_supertrend(high, low, close, period=10, multiplier=3.0):
    """Calculate Supertrend indicator"""
    n = len(close)
    if n < period:
        return np.zeros(n), np.zeros(n)
    
    atr = calculate_atr(high, low, close, period)
    
    supertrend = np.zeros(n)
    trend_direction = np.ones(n)
    
    upper_band = (high + low) / 2 + multiplier * atr
    lower_band = (high + low) / 2 - multiplier * atr
    
    supertrend[period] = lower_band[period]
    
    for i in range(period + 1, n):
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


def calculate_zscore(close, period=20):
    """Calculate Z-score for mean reversion filter"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    zscore = np.zeros(n)
    for i in range(period - 1, n):
        window = close[i - period + 1:i + 1]
        mean = np.mean(window)
        std = np.std(window)
        if std > 0:
            zscore[i] = (close[i] - mean) / std
        else:
            zscore[i] = 0.0
    
    return zscore


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Initialize all arrays
    signals = np.zeros(n)
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    kama_1h = calculate_kama(close, period=10)
    supertrend_1h, st_direction_1h = calculate_supertrend(high, low, close, period=10, multiplier=3.0)
    zscore_1h = calculate_zscore(close, period=20)
    
    # 4h trend filters using mtf_data helper (MANDATORY - actual Binance 4h candles)
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
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.25
    SIZE_HALF = 0.125
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 55
    RSI_SHORT_MIN = 45
    RSI_SHORT_MAX = 60
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.0
    
    # Z-score filter (avoid extremes)
    ZSCORE_MAX = 2.0
    
    first_valid = 100
    
    # Track position state (simplified)
    in_position = False
    position_side = 0
    entry_price = 0.0
    tp_triggered = False
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(first_valid, n):
        # Skip invalid data
        if np.isnan(atr_1h[i]) or atr_1h[i] == 0 or np.isnan(kama_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # 4h trend direction
        trend_4h = 0
        if kama_4h_aligned[i] > 0:
            if close[i] > kama_4h_aligned[i]:
                trend_4h = 1
            elif close[i] < kama_4h_aligned[i]:
                trend_4h = -1
        
        st_trend_4h = st_direction_4h_aligned[i]
        
        # 1h indicators
        rsi_val = rsi_1h[i]
        st_direction_1h = st_direction_1h[i]
        atr = atr_1h[i]
        price = close[i]
        zscore = zscore_1h[i]
        
        # Exit logic for existing positions
        if in_position:
            # Update highest/lowest since entry
            if position_side == 1:
                highest_since_entry = max(highest_since_entry, price)
            else:
                lowest_since_entry = min(lowest_since_entry, price)
            
            # Stoploss check (2.0*ATR)
            if position_side == 1:
                stoploss_price = entry_price - ATR_STOP_MULT * atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
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
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
                
            elif position_side == -1:
                stoploss_price = entry_price + ATR_STOP_MULT * atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    in_position = False
                    position_side = 0
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
                        in_position = False
                        position_side = 0
                        entry_price = 0.0
                        tp_triggered = False
                        highest_since_entry = 0.0
                        lowest_since_entry = 0.0
                        continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1] if i > 0 else 0.0
            continue
        
        # Z-score filter (avoid entering at extremes)
        if abs(zscore) > ZSCORE_MAX:
            signals[i] = 0.0
            continue
        
        # Entry logic: 4h trend + 1h RSI pullback + 1h Supertrend confirmation
        if trend_4h == 1 and st_trend_4h == 1:  # Bullish trend on 4h
            # Wait for RSI pullback on 1h, confirm with 1h Supertrend
            if (RSI_LONG_MIN <= rsi_val <= RSI_LONG_MAX and 
                st_direction_1h == 1):
                signals[i] = SIZE_FULL
                in_position = True
                position_side = 1
                entry_price = price
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
                
        elif trend_4h == -1 and st_trend_4h == -1:  # Bearish trend on 4h
            # Wait for RSI pullback on 1h, confirm with 1h Supertrend
            if (RSI_SHORT_MIN <= rsi_val <= RSI_SHORT_MAX and 
                st_direction_1h == -1):
                signals[i] = -SIZE_FULL
                in_position = True
                position_side = -1
                entry_price = price
                tp_triggered = False
                highest_since_entry = price
                lowest_since_entry = price
        
        else:
            signals[i] = 0.0
    
    return signals