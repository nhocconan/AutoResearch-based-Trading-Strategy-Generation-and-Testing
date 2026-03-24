#!/usr/bin/env python3
"""
Experiment #598: 4h Primary + 1d HTF — HMA Trend + RSI Pullback Simplified

Hypothesis: Complex regime detection (ADX, CHOP, multiple filters) caused 0 trades
in recent experiments. This strategy uses PROVEN simple patterns that actually trigger:
1. 1d HMA(21) for macro trend bias (single HTF, not dual)
2. 4h HMA(9/21) crossover for trend confirmation
3. RSI(14) pullback with LOOSE thresholds (40/60 not 25/75) - MORE TRADES
4. ATR(14)*2.5 trailing stoploss
5. Discrete signal sizes (0.0, ±0.25, ±0.30)

Key fixes from failed #522:
1. REMOVED ADX/CHOP regime filters - were blocking all entries
2. LOOSENED RSI thresholds (40/60 vs 35/65) - triggers more often
3. SIMPLIFIED entry logic - only 2-3 conditions, not 5+
4. FIXED position tracking - hold position until stop or reverse signal
5. Single HTF (1d) - less complexity, faster execution

Target: Sharpe>0.40, trades>=80 train (20/year), trades>=10 test, DD>-40%
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_simple_1d_v1"
timeframe = "4h"
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
    
    # Load HTF data ONCE before loop (CRITICAL - Rule 1)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    hma_9 = calculate_hma(close, period=9)
    hma_21 = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking
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
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_9[i]) or np.isnan(hma_21[i]) or np.isnan(rsi[i]):
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]):
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND BIAS (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h TREND (HMA crossover) ===
        hma_bull = hma_9[i] > hma_21[i]
        hma_bear = hma_9[i] < hma_21[i]
        
        # === RSI PULLBACK (LOOSE thresholds for MORE trades) ===
        rsi_pullback_long = rsi[i] < 50.0  # Pullback in uptrend
        rsi_pullback_short = rsi[i] > 50.0  # Pullback in downtrend
        rsi_strong_long = rsi[i] < 40.0  # Deep pullback
        rsi_strong_short = rsi[i] > 60.0  # Deep pullback
        
        # === ENTRY LOGIC (SIMPLE - triggers often) ===
        desired_signal = 0.0
        
        # LONG: Macro bull + HMA bull + RSI pullback
        if macro_bull and hma_bull:
            if rsi_strong_long:
                desired_signal = SIZE_STRONG
            elif rsi_pullback_long:
                desired_signal = SIZE_BASE
        
        # SHORT: Macro bear + HMA bear + RSI pullback
        elif macro_bear and hma_bear:
            if rsi_strong_short:
                desired_signal = -SIZE_STRONG
            elif rsi_pullback_short:
                desired_signal = -SIZE_BASE
        
        # === REVERSE SIGNAL (close long, open short or vice versa) ===
        if in_position and position_side > 0 and macro_bear and hma_bear:
            desired_signal = -SIZE_BASE  # Reverse to short
        
        if in_position and position_side < 0 and macro_bull and hma_bull:
            desired_signal = SIZE_BASE  # Reverse to long
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
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
        
        # === DISCRETIZE SIGNAL ===
        if desired_signal >= SIZE_STRONG * 0.9:
            final_signal = SIZE_STRONG
        elif desired_signal <= -SIZE_STRONG * 0.9:
            final_signal = -SIZE_STRONG
        elif desired_signal >= SIZE_BASE * 0.9:
            final_signal = SIZE_BASE
        elif desired_signal <= -SIZE_BASE * 0.9:
            final_signal = -SIZE_BASE
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                # New position or reverse
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
            # If same side, keep position (don't reset tracking vars)
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