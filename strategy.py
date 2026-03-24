#!/usr/bin/env python3
"""
Experiment #1596: 12h Primary + 1d HTF — Simplified HMA Trend Strategy

Hypothesis: After 11 failed 12h/4h experiments with complex regime switching and 0 trades,
simplicity is key. The problem isn't the indicators - it's entry conditions too strict.

Key learnings from failures:
- #1586, #1592, #1593: 0 trades (entry conditions too strict)
- Complex regime filters prevent signals from triggering
- RSI ranges like 45-55 are too narrow for 12h timeframe

This strategy uses:
1. 12h HMA(16/48) crossover - proven trend signal with less lag than EMA
2. 1d HMA(21) for trend bias (only trade with daily trend)
3. RSI(14) 35-65 filter - WIDE range to allow signals (not 45-55!)
4. ATR(14) 2.5x trailing stop for drawdown control
5. Discrete position sizing (0.25) to minimize fee churn

Why this should generate trades (>10/year target):
- HMA crossover triggers on clear trend changes (not rare events)
- RSI 35-65 is wide enough to allow entries in both bull/bear
- 12h timeframe = ~730 bars/year, HMA crossover ~20-40 times/year
- Simple logic = more reliable signals = meets trade minimum

Timeframe: 12h (required for this experiment)
HTF: 1d HMA for bias (use mtf_data helper - call ONCE before loop)
Target: Sharpe > 0.618, trades > 10/symbol train, > 3/symbol test, DD > -50%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_hma_crossover_1d_trend_rsi_atr_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        if w_period < 1:
            return np.full(len(data), np.nan)
        result = np.full(len(data), np.nan)
        weights = np.arange(1, w_period + 1, dtype=float)
        weights /= weights.sum()
        for i in range(w_period - 1, len(data)):
            window = data[i - w_period + 1:i + 1]
            if np.any(np.isnan(window)):
                continue
            result[i] = np.sum(window * weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan)
    mask = ~np.isnan(wma_half) & ~np.isnan(wma_full)
    diff[mask] = 2.0 * wma_half[mask] - wma_full[mask]
    
    hma = wma(diff, sqrt_period)
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index with proper min_periods"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    gain_smooth = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    loss_smooth = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan)
    mask = loss_smooth > 1e-10
    rsi[mask] = 100.0 - (100.0 / (1.0 + gain_smooth[mask] / loss_smooth[mask]))
    rsi[loss_smooth <= 1e-10] = 100.0
    rsi[:period] = np.nan
    
    return rsi

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

def calculate_sma(close, period=200):
    """Simple Moving Average with proper min_periods"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = np.full(n, np.nan)
    for i in range(period - 1, n):
        sma[i] = np.mean(close[i-period+1:i+1])
    
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (12h) indicators
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    # HMA crossover signals (16/48) - proven combination
    hma_fast = calculate_hma(close, period=16)
    hma_slow = calculate_hma(close, period=48)
    
    # SMA 200 for long-term trend filter
    sma_200 = calculate_sma(close, period=200)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track previous HMA relationship for crossover detection
    prev_hma_fast = np.nan
    prev_hma_slow = np.nan
    
    for i in range(200, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_hma_fast = hma_fast[i] if not np.isnan(hma_fast[i]) else prev_hma_fast
            prev_hma_slow = hma_slow[i] if not np.isnan(hma_slow[i]) else prev_hma_slow
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_hma_fast = hma_fast[i] if not np.isnan(hma_fast[i]) else prev_hma_fast
            prev_hma_slow = hma_slow[i] if not np.isnan(hma_slow[i]) else prev_hma_slow
            continue
        
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_hma_fast = hma_fast[i] if not np.isnan(hma_fast[i]) else prev_hma_fast
            prev_hma_slow = hma_slow[i] if not np.isnan(hma_slow[i]) else prev_hma_slow
            continue
        
        # === TREND BIAS (1d HMA) ===
        daily_bull = close[i] > hma_1d_aligned[i]
        daily_bear = close[i] < hma_1d_aligned[i]
        
        # === HMA CROSSOVER SIGNAL ===
        hma_bull = hma_fast[i] > hma_slow[i]
        hma_bear = hma_fast[i] < hma_slow[i]
        
        # Detect actual crossover (not just position)
        hma_cross_up = False
        hma_cross_down = False
        
        if not np.isnan(prev_hma_fast) and not np.isnan(prev_hma_slow):
            hma_cross_up = (prev_hma_fast <= prev_hma_slow) and (hma_fast[i] > hma_slow[i])
            hma_cross_down = (prev_hma_fast >= prev_hma_slow) and (hma_fast[i] < hma_slow[i])
        
        # === RSI FILTER (WIDE RANGE to allow signals) ===
        # 35-65 is wide enough for 12h timeframe (not 45-55!)
        rsi_ok_long = rsi[i] >= 35.0
        rsi_ok_short = rsi[i] <= 65.0
        
        # === SMA 200 FILTER (optional trend confirmation) ===
        above_sma200 = close[i] > sma_200[i] if not np.isnan(sma_200[i]) else True
        below_sma200 = close[i] < sma_200[i] if not np.isnan(sma_200[i]) else True
        
        # === PRIMARY SIGNAL ===
        desired_signal = 0.0
        
        # LONG: HMA cross up OR (HMA bull + Daily bull) + RSI support
        # Use crossover for entry, hold on trend continuation
        if hma_cross_up and daily_bull and rsi_ok_long:
            desired_signal = BASE_SIZE
        elif hma_bull and daily_bull and rsi_ok_long and in_position and position_side > 0:
            desired_signal = BASE_SIZE
        
        # SHORT: HMA cross down OR (HMA bear + Daily bear) + RSI support
        elif hma_cross_down and daily_bear and rsi_ok_short:
            desired_signal = -BASE_SIZE
        elif hma_bear and daily_bear and rsi_ok_short and in_position and position_side < 0:
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
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(final_signal) != position_side:
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif position_side > 0:
                highest_since_entry = max(highest_since_entry, close[i])
            elif position_side < 0:
                lowest_since_entry = min(lowest_since_entry, close[i])
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_price = 0.0
                entry_atr = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
        
        # Update previous HMA values for next iteration
        prev_hma_fast = hma_fast[i]
        prev_hma_slow = hma_slow[i]
        
        signals[i] = final_signal
    
    return signals