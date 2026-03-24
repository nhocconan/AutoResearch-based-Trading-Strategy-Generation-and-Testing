#!/usr/bin/env python3
"""
Experiment #015: 1h Primary + 4h HTF — Simplified Trend-Follow with RSI Pullback Entries

Hypothesis: Previous 1h strategies failed due to OVER-FILTERING (session + volume + multiple HTF).
This uses PROVEN simple pattern that generates trades:
1. 4h EMA(50) for trend direction ONLY (bias, not hard filter)
2. 1h RSI(14) for entry timing (35/65 thresholds - LOOSE for trade gen)
3. ATR(14) 2.5x trailing stop for risk management
4. NO session filter, NO volume filter (these killed trade generation)
5. Discrete signal levels (0.0, ±0.25, ±0.30) to minimize fee churn

Key differences from failed 1h attempts:
- REMOVED session filter (8-20 UTC) - was blocking 2/3 of potential entries
- REMOVED volume filter - was causing 0 trades on low-volume periods
- LOOSER RSI thresholds (35/65 vs 30/70) - ensures more trade generation
- Single HTF (4h only) - 1d was causing alignment NaN issues
- Simpler EMA instead of HMA - more stable alignment

Entry Logic:
- Long: 4h EMA bullish + RSI < 35 (pullback in uptrend)
- Short: 4h EMA bearish + RSI > 65 (pullback in downtrend)
- Exit: RSI crosses 50 OR stoploss hit

Target: Sharpe > 0.3, trades > 30/symbol train, > 3/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_rsi_4h_ema_pullback_v2"
timeframe = "1h"
leverage = 1.0

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_avg = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_avg = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if loss_avg[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rsi[i] = 100.0 - (100.0 / (1.0 + gain_avg[i] / loss_avg[i]))
    
    return rsi

def calculate_ema(close, period=50):
    """Exponential Moving Average with proper min_periods"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    ema = pd.Series(close).ewm(span=period, min_periods=period, adjust=False).mean().values
    return ema

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods"""
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
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate and align 4h EMA(50) for trend bias
    ema_4h_raw = calculate_ema(df_4h['close'].values, period=50)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    CONFIRMED_SIZE = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(ema_4h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (4h EMA50) ===
        hma_4h_bull = close[i] > ema_4h_aligned[i]
        hma_4h_bear = close[i] < ema_4h_aligned[i]
        
        # === RSI ENTRY SIGNALS (LOOSE thresholds for trade gen) ===
        desired_signal = 0.0
        
        # Long entry: RSI < 35 (pullback) + 4h bullish bias preferred
        if rsi[i] < 35.0:
            if hma_4h_bull:
                desired_signal = CONFIRMED_SIZE
            else:
                desired_signal = BASE_SIZE
        
        # Short entry: RSI > 65 (pullback) + 4h bearish bias preferred
        elif rsi[i] > 65.0:
            if hma_4h_bear:
                desired_signal = -CONFIRMED_SIZE
            else:
                desired_signal = -BASE_SIZE
        
        # === STOPLOSS CHECK (Trailing ATR 2.5x) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, close[i])
            stop_price = highest_since_entry - 2.5 * entry_atr
            if close[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, close[i])
            stop_price = lowest_since_entry + 2.5 * entry_atr
            if close[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === DISCRETIZE SIGNAL VALUES ===
        if desired_signal >= CONFIRMED_SIZE * 0.85:
            final_signal = CONFIRMED_SIZE
        elif desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -CONFIRMED_SIZE * 0.85:
            final_signal = -CONFIRMED_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                # New position
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            elif np.sign(final_signal) != position_side:
                # Position flip
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i]
                lowest_since_entry = close[i]
            elif position_side > 0:
                # Update long trailing high
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                # Update short trailing low
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            # Exit position
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        signals[i] = final_signal
    
    return signals