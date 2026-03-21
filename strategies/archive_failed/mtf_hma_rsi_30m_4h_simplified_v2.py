#!/usr/bin/env python3
"""
EXPERIMENT #022 - Simplified MTF HMA+RSI (30m+4h v2)
==================================================================================================
Hypothesis: Simplify the MTF approach from 3 timeframes to 2 (30m base + 4h trend).
The current best uses 15m+1h+4h but has complex state tracking that may cause bugs.
This version:
- Uses 30m base (cleaner signals than 15m, more trades than 1h)
- 4h HMA trend filter only (remove 1h complexity)
- RSI pullback entries on 30m
- Simpler position management without complex state tracking
- Discrete position sizes (0.0, ±0.25, ±0.35)

Why this should work:
- Fewer timeframes = fewer alignment issues
- 30m has proven success in other experiments
- Simpler logic = fewer bugs in signal generation
- HMA trend filter is proven (current best uses it)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_hma_rsi_30m_4h_simplified_v2"
timeframe = "30m"
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


def generate_signals(prices: pd.DataFrame) -> np.ndarray:
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # 30m indicators for entry timing
    atr_30m = calculate_atr(high, low, close, period=14)
    rsi_30m = calculate_rsi(close, period=14)
    hma_30m = calculate_hma(close, period=21)
    
    # Get 4h data using mtf_data helper for trend filter
    try:
        df_4h = get_htf_data(prices, '4h')
        c_4h = df_4h['close'].values
        h_4h = df_4h['high'].values
        l_4h = df_4h['low'].values
        
        # 4h HMA for trend direction
        hma_4h = calculate_hma(c_4h, period=21)
        
        # 4h Supertrend for trend confirmation
        _, st_direction_4h = calculate_supertrend(h_4h, l_4h, c_4h, period=10, multiplier=3.0)
        
        # Align 4h indicators to 30m timeframe
        hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
        st_direction_4h_aligned = align_htf_to_ltf(prices, df_4h, st_direction_4h)
        
    except Exception:
        hma_4h_aligned = np.zeros(n)
        st_direction_4h_aligned = np.zeros(n)
    
    # Generate signals with multi-timeframe logic
    signals = np.zeros(n)
    
    # Position sizing - DISCRETE levels (CRITICAL for drawdown control)
    SIZE_FULL = 0.35
    SIZE_HALF = 0.175
    SIZE_QUARTER = 0.0875
    
    # RSI thresholds for pullback entries
    RSI_LONG_ENTRY = 45
    RSI_SHORT_ENTRY = 55
    RSI_EXIT_LONG = 65
    RSI_EXIT_SHORT = 35
    
    # ATR stoploss multiplier
    ATR_STOP_MULT = 2.5
    
    first_valid = max(200, 14 * 2, 21)
    
    # Track position state
    position_side = np.zeros(n)
    entry_price = np.zeros(n)
    highest_since_entry = np.zeros(n)
    lowest_since_entry = np.zeros(n)
    
    for i in range(first_valid, n):
        if np.isnan(atr_30m[i]) or np.isnan(rsi_30m[i]) or atr_30m[i] == 0:
            signals[i] = 0.0
            continue
        
        # Get aligned 4h values
        hma_4h_val = hma_4h_aligned[i] if i < len(hma_4h_aligned) else 0
        st_4h_val = st_direction_4h_aligned[i] if i < len(st_direction_4h_aligned) else 0
        
        # Determine 4h trend direction
        trend_4h = 0
        if hma_4h_val > 0 and i >= first_valid:
            # Get corresponding 4h close for price vs HMA comparison
            idx_4h = min(i // 8, len(hma_4h_aligned) - 1)  # 8 x 30m = 4h
            if idx_4h >= 0 and idx_4h < len(c_4h):
                if c_4h[idx_4h] > hma_4h[idx_4h]:
                    trend_4h = 1
                elif c_4h[idx_4h] < hma_4h[idx_4h]:
                    trend_4h = -1
        
        # Check existing positions first (stoploss and take profit)
        if position_side[i - 1] != 0:
            prev_side = position_side[i - 1]
            prev_entry = entry_price[i - 1] if entry_price[i - 1] > 0 else close[i - 1]
            prev_high = highest_since_entry[i - 1] if highest_since_entry[i - 1] > 0 else prev_entry
            prev_low = lowest_since_entry[i - 1] if lowest_since_entry[i - 1] > 0 else prev_entry
            
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
            
            # Stoploss check (2.5*ATR)
            if prev_side == 1:
                stoploss_price = prev_entry - ATR_STOP_MULT * atr_30m[i]
                if price < stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit at 2R - reduce to half
                tp_price = prev_entry + 2 * ATR_STOP_MULT * atr_30m[i]
                if price >= tp_price and signals[i - 1] == SIZE_FULL:
                    signals[i] = SIZE_HALF
                    position_side[i] = 1
                    entry_price[i] = prev_entry
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R after TP
                if signals[i - 1] == SIZE_HALF:
                    trail_stop = current_high - ATR_STOP_MULT * atr_30m[i]
                    if price < trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # RSI exit signal
                if rsi_30m[i] >= RSI_EXIT_LONG:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                    
            elif prev_side == -1:
                stoploss_price = prev_entry + ATR_STOP_MULT * atr_30m[i]
                if price > stoploss_price:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
                
                # Take profit at 2R - reduce to half
                tp_price = prev_entry - 2 * ATR_STOP_MULT * atr_30m[i]
                if price <= tp_price and signals[i - 1] == -SIZE_FULL:
                    signals[i] = -SIZE_HALF
                    position_side[i] = -1
                    entry_price[i] = prev_entry
                    highest_since_entry[i] = current_high
                    lowest_since_entry[i] = current_low
                    continue
                
                # Trail stop at 1R after TP
                if signals[i - 1] == -SIZE_HALF:
                    trail_stop = current_low + ATR_STOP_MULT * atr_30m[i]
                    if price > trail_stop:
                        signals[i] = 0.0
                        position_side[i] = 0
                        entry_price[i] = 0
                        highest_since_entry[i] = 0
                        lowest_since_entry[i] = 0
                        continue
                
                # RSI exit signal
                if rsi_30m[i] <= RSI_EXIT_SHORT:
                    signals[i] = 0.0
                    position_side[i] = 0
                    entry_price[i] = 0
                    highest_since_entry[i] = 0
                    lowest_since_entry[i] = 0
                    continue
            
            # Hold position if no exit triggered
            signals[i] = signals[i - 1]
            position_side[i] = position_side[i - 1]
            entry_price[i] = entry_price[i - 1]
            highest_since_entry[i] = highest_since_entry[i - 1]
            lowest_since_entry[i] = lowest_since_entry[i - 1]
            continue
        
        # Entry logic: 4h trend + 30m RSI pullback
        price = close[i]
        
        # Long entry: 4h bullish + RSI pullback on 30m
        if trend_4h == 1 and st_4h_val == 1:
            if rsi_30m[i] <= RSI_LONG_ENTRY and rsi_30m[i] >= 30:
                signals[i] = SIZE_FULL
                position_side[i] = 1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
                
        # Short entry: 4h bearish + RSI pullback on 30m
        elif trend_4h == -1 and st_4h_val == -1:
            if rsi_30m[i] >= RSI_SHORT_ENTRY and rsi_30m[i] <= 70:
                signals[i] = -SIZE_FULL
                position_side[i] = -1
                entry_price[i] = price
                highest_since_entry[i] = price
                lowest_since_entry[i] = price
        
        else:
            signals[i] = 0.0
            position_side[i] = 0
    
    return signals