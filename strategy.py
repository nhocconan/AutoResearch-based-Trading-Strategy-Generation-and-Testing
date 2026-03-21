#!/usr/bin/env python3
"""
EXPERIMENT #025 - Simplified HMA+RSI on 1h with 4h trend filter
==================================================================================================
Hypothesis: The current best (30m+4h) works well but is still complex. Moving to 1h base 
timeframe should provide cleaner signals with less noise while maintaining trade frequency.
Key simplifications:
- Remove MACD, BBW, Supertrend complexity
- Use only HMA trend + RSI pullback (proven in #022)
- Dynamic ATR-based position sizing (reduce size in high volatility)
- Tighter stoploss at 1.5*ATR to reduce drawdown
- Discrete signal levels: 0.0, ±0.25, ±0.35

Why 1h instead of 30m:
- Less noise and whipsaws than 30m
- Still generates sufficient trades (>10)
- Better alignment with 4h HTF data (4x ratio vs 8x for 30m)
- Proven success in similar strategies

Position sizing: base 0.30, scaled by ATR volatility (reduce size when ATR is high)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_1h_4h_simplified_v3"
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


def calculate_hma(close, period=21):
    """Calculate Hull Moving Average"""
    n = len(close)
    if n < period:
        return np.zeros(n)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half_period, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, adjust=False).mean().values
    
    hma = pd.Series(2 * wma1 - wma2).ewm(span=sqrt_period, adjust=False).mean().values
    
    return hma


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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 1h indicators for entry timing
    atr_1h = calculate_atr(high, low, close, period=14)
    rsi_1h = calculate_rsi(close, period=14)
    hma_1h_fast = calculate_hma(close, period=16)
    hma_1h_slow = calculate_hma(close, period=48)
    
    # Get 4h data using mtf_data helper (MUST use this for proper alignment)
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_4h = calculate_hma(c_4h, period=21)
        
        # Align 4h indicators to 1h timeframe (auto shift for completed bars)
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        c_4h_aligned = align_htf_to_ltf(prices, df_4h, c_4h)
        
    except Exception:
        hma_4h_aligned = np.zeros(n)
        c_4h_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    BASE_SIZE = 0.30
    SIZE_HALF = 0.15
    MAX_SIZE = 0.35
    MIN_SIZE = 0.20
    
    # ATR-based dynamic sizing parameters
    TARGET_ATR_PCT = 0.02  # Target 2% ATR relative to price
    ATR_LOOKBACK = 50  # Lookback for ATR percentile
    
    # RSI thresholds for pullback entries
    RSI_LONG_MIN = 40
    RSI_LONG_MAX = 60
    RSI_SHORT_MIN = 40
    RSI_SHORT_MAX = 60
    
    # ATR stoploss multiplier (tighter than before)
    ATR_STOP_MULT = 1.5
    ATR_TP_MULT = 3.0  # 2R take profit
    
    first_valid = max(100, 48 * 2, 21 * 2)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    tp_triggered = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    entry_atr = np.zeros(n)
    
    for i in range(first_valid, n):
        # Validate data
        if np.isnan(atr_1h[i]) or np.isnan(rsi_1h[i]) or atr_1h[i] == 0:
            signals[i] = 0.0
            if i > 0:
                position_side[i] = 0
            continue
        
        # Get aligned 4h values
        hma_4h_val = hma_4h_aligned[i] if i < len(hma_4h_aligned) else 0
        c_4h_val = c_4h_aligned[i] if i < len(c_4h_aligned) else close[i]
        
        # Calculate 4h trend direction (price vs HMA)
        trend_4h = 0
        if hma_4h_val > 0 and c_4h_val > 0:
            if c_4h_val > hma_4h_val * 1.002:  # 0.2% buffer
                trend_4h = 1
            elif c_4h_val < hma_4h_val * 0.998:
                trend_4h = -1
        
        # Calculate dynamic position size based on ATR volatility
        atr_pct = atr_1h[i] / close[i] if close[i] > 0 else 0.02
        
        # Calculate ATR percentile over lookback
        if i >= ATR_LOOKBACK:
            atr_history = atr_1h[i-ATR_LOOKBACK:i] / close[i-ATR_LOOKBACK:i]
            atr_percentile = np.percentile(atr_history, 50)
        else:
            atr_percentile = atr_pct
        
        # Scale position size inversely with volatility
        vol_scale = TARGET_ATR_PCT / atr_percentile if atr_percentile > 0 else 1.0
        vol_scale = np.clip(vol_scale, 0.5, 1.5)  # Limit scaling range
        
        SIZE_FULL = np.clip(BASE_SIZE * vol_scale, MIN_SIZE, MAX_SIZE)
        SIZE_HALF_CURRENT = SIZE_FULL / 2
        
        # Check stoploss and take profit for existing positions FIRST
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_tp = tp_triggered[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            prev_entry_atr = entry_atr[i - 1] if entry_atr[i - 1] > 0 else atr_1h[i - 1]
            
            price = close[i]
            
            # Update highest/lowest since entry
            if prev_side == 1:
                current_high = max(prev_high, price)
                current_low = min(prev_low, price) if prev_low > 0 else price
            else:
                current_high = max(prev_high, price) if prev_high > 0 else price
                current_low = min(prev_low, price)
            
            highest_since_entry[i] = current_high
            lowest_since_entry[i] = current_low
            
            # Stoploss check (1.5*ATR from entry)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * prev_entry_atr
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    entry_atr[i] = 0
                    continue
                
                # Take profit check (3R = 2R profit) - reduce to half
                tp_price = prev_entry + ATR_TP_MULT * prev_entry_atr
                if not prev_tp and price >= tp_price:
                    signals[i] = SIZE_HALF_CURRENT
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    entry_atr[i] = prev_entry_atr
                    continue
                
                # Trail stop at 1.5R profit (breakeven + profit)
                if prev_tp:
                    trail_stop = current_high - ATR_STOP_MULT * prev_entry_atr
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        entry_atr[i] = 0
                        continue
            
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * prev_entry_atr
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    tp_triggered[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    entry_atr[i] = 0
                    continue
                
                # Take profit check (3R) - reduce to half
                tp_price = prev_entry - ATR_TP_MULT * prev_entry_atr
                if not prev_tp and price <= tp_price:
                    signals[i] = -SIZE_HALF_CURRENT
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    tp_triggered[i] = 1
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    entry_atr[i] = prev_entry_atr
                    continue
                
                # Trail stop at 1.5R profit
                if prev_tp:
                    trail_stop = current_low + ATR_STOP_MULT * prev_entry_atr
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        tp_triggered[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        entry_atr[i] = 0
                        continue
            
            # Check if 4h trend changed - close position
            if prev_side == 1 and trend_4h == -1:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                entry_atr[i] = 0
                continue
            elif prev_side == -1 and trend_4h == 1:
                signals[i] = 0.0
                position_side[i] = 0
                entry_price[i] = 0
                tp_triggered[i] = 0
                highest_since_entry[i] = 0
                lowest_since_entry[i] = 0
                entry_atr[i] = 0
                continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            tp_triggered[i] = tp_triggered[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            entry_atr[i] = entry_atr[i - 1]
            continue
        
        # Entry logic: 4h trend + 1h HMA crossover + 1h RSI pullback
        price = close[i]
        
        # Check HMA crossover on 1h
        hma_cross_bullish = hma_1h_fast[i] > hma_1h_slow[i] and hma_1h_fast[i-1] <= hma_1h_slow[i-1]
        hma_cross_bearish = hma_1h_fast[i] < hma_1h_slow[i] and hma_1h_fast[i-1] >= hma_1h_slow[i-1]
        
        # HMA trend alignment (fast above slow for long, below for short)
        hma_trend_bullish = hma_1h_fast[i] > hma_1h_slow[i]
        hma_trend_bearish = hma_1h_fast[i] < hma_1h_slow[i]
        
        if trend_4h == 1 and hma_trend_bullish:  # Bullish trend on 4h + 1h
            # RSI pullback on 1h (not overbought, in neutral zone)
            if RSI_LONG_MIN <= rsi_1h[i] <= RSI_LONG_MAX:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                entry_atr[i] = atr_1h[i]
                
        elif trend_4h == -1 and hma_trend_bearish:  # Bearish trend on 4h + 1h
            # RSI pullback on 1h (not oversold, in neutral zone)
            if RSI_SHORT_MIN <= rsi_1h[i] <= RSI_SHORT_MAX:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                tp_triggered[i] = 0
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                entry_atr[i] = atr_1h[i]
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals