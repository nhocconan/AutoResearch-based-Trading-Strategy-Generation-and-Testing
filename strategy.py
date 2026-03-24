#!/usr/bin/env python3
"""
Experiment #567: 6h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Previous 6h experiments failed with Sharpe=0.000 (zero trades) because
entry conditions were too strict (multiple regime filters, ADX+CHOP+RSI all required).
This strategy SIMPLIFIES entry logic to ensure trades generate while maintaining
quality through HTF trend alignment.

Key changes from failed experiments:
1. REMOVED ADX/CHOP regime filters (caused 0 trades in #555, #560, #563)
2. Simplified to HMA crossover + RSI pullback (proven pattern from best 4h strategies)
3. LOOSER RSI thresholds (40-60 instead of 25-75) to catch more trend continuations
4. 1d HMA(21) for macro bias only - don't require perfect alignment
5. ATR stoploss at 2.5x mandatory on all positions

Strategy logic:
1. 1d HMA(21) = macro trend bias (aligned via mtf_data helper)
2. 6h HMA(9) = fast trend following
3. 6h HMA(21) = slow trend confirmation
4. Entry: Fast HMA > Slow HMA + price > 1d HMA + RSI pullback (40-55) for long
5. Entry: Fast HMA < Slow HMA + price < 1d HMA + RSI pullback (45-60) for short
6. Stoploss: 2.5x ATR trailing from entry
7. Size: 0.30 for trend-aligned, 0.20 for weaker signals

Target: 40-80 trades/year on 6h (fewer than 4h due to higher TF)
Timeframe: 6h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_hma_rsi_pullback_1d_simple_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - faster response than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma1 = pd.Series(close).ewm(span=half, min_periods=half, adjust=False).mean().values
    wma2 = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    diff = 2.0 * wma1 - wma2
    hma = pd.Series(diff).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    gain = np.concatenate([[0.0], gain])
    loss = np.concatenate([[0.0], loss])
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.zeros(n)
    rsi[:] = np.nan
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 6h indicators
    hma_fast = calculate_hma(close, period=9)
    hma_slow = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF MACRO BIAS (1d HMA) ===
        htf_bull = close[i] > hma_1d_aligned[i]
        htf_bear = close[i] < hma_1d_aligned[i]
        
        # === 6h TREND (HMA crossover) ===
        trend_bull = hma_fast[i] > hma_slow[i]
        trend_bear = hma_fast[i] < hma_slow[i]
        
        # HMA slope confirmation
        hma_fast_slope_bull = hma_fast[i] > hma_fast[i-3] if i >= 3 and not np.isnan(hma_fast[i-3]) else False
        hma_fast_slope_bear = hma_fast[i] < hma_fast[i-3] if i >= 3 and not np.isnan(hma_fast[i-3]) else False
        
        # === RSI PULLBACK (not extreme - catch trend continuations) ===
        rsi_pullback_long = rsi[i] >= 40.0 and rsi[i] <= 55.0
        rsi_pullback_short = rsi[i] >= 45.0 and rsi[i] <= 60.0
        rsi_recovery_long = rsi[i] > rsi[i-1] if i > 0 else False
        rsi_recovery_short = rsi[i] < rsi[i-1] if i > 0 else False
        
        # === ENTRY LOGIC (SIMPLIFIED - LOOSER CONDITIONS) ===
        desired_signal = 0.0
        
        # LONG entries - trend aligned with HTF
        if htf_bull and trend_bull:
            if rsi_pullback_long:
                desired_signal = SIZE_STRONG
            elif rsi[i] < 45.0 and rsi_recovery_long:
                desired_signal = SIZE_BASE
            elif hma_fast_slope_bull and rsi[i] > 50.0:
                desired_signal = SIZE_BASE
        
        # SHORT entries - trend aligned with HTF
        elif htf_bear and trend_bear:
            if rsi_pullback_short:
                desired_signal = -SIZE_STRONG
            elif rsi[i] > 55.0 and rsi_recovery_short:
                desired_signal = -SIZE_BASE
            elif hma_fast_slope_bear and rsi[i] < 50.0:
                desired_signal = -SIZE_BASE
        
        # Weaker signals when HTF and 6h disagree (reduce size)
        elif trend_bull and rsi_pullback_long:
            desired_signal = SIZE_BASE * 0.7
        elif trend_bear and rsi_pullback_short:
            desired_signal = -SIZE_BASE * 0.7
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            if low[i] < stop_price:
                stoploss_triggered = True
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            if high[i] > stop_price:
                stoploss_triggered = True
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        elif abs(desired_signal) >= SIZE_BASE * 0.5:
            final_signal = np.sign(desired_signal) * SIZE_BASE * 0.7
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = final_signal
    
    return signals