#!/usr/bin/env python3
"""
Experiment #198: 4h Primary + 1d HTF — Simplified Trend Pullback with Regime Bias

Hypothesis: Experiments #189, #190, #193, #196, #197 failed with 0 trades due to 
overly strict filter combinations. This strategy SIMPLIFIES while maintaining 
multi-timeframe edge:

1. 1d HMA(50) = Regime bias (weights position size, NOT hard filter)
2. 4h HMA(21/48) crossover = Trend direction
3. RSI(14) pullback = Entry timing (35-65 thresholds for more trades)
4. ATR(14) trailing stop = 2.5x ATR

Key improvements from failures:
- 1d HMA is BIAS not filter (allows trades in both directions)
- RSI thresholds widened to 35-65 (not 20-80) for more entries
- HMA crossover (fast/slow) instead of price vs HMA (more responsive)
- No choppiness/ADX filters that killed trade frequency
- Target 40-60 trades/year on 4h timeframe

Position sizing: 0.25 base, 0.35 when HTF confirms
Stoploss: 2.5x ATR trailing

Target: Sharpe>0.399 (beat current best), DD>-40%, trades>=30 train, trades>=3 test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_cross_rsi_pullback_1d_v1"
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
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA for regime bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_fast = calculate_hma(close, period=16)  # Fast HMA
    hma_slow = calculate_hma(close, period=48)  # Slow HMA
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # 25% base position
    SIZE_CONFIRMED = 0.35  # 35% when HTF confirms
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            continue
        if np.isnan(hma_fast[i]) or np.isnan(hma_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            continue
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            continue
        
        # === HTF REGIME BIAS (1d HMA) ===
        # Not a hard filter - just adjusts position size
        htf_bullish = not np.isnan(hma_1d_aligned[i]) and close[i] > hma_1d_aligned[i]
        htf_bearish = not np.isnan(hma_1d_aligned[i]) and close[i] < hma_1d_aligned[i]
        
        # === 4h HMA CROSSOVER TREND ===
        hma_bullish = hma_fast[i] > hma_slow[i]
        hma_bearish = hma_fast[i] < hma_slow[i]
        
        # === HMA SLOPE CONFIRMATION ===
        hma_slope_bull = False
        hma_slope_bear = False
        if i >= 5 and not np.isnan(hma_slow[i-5]):
            hma_slope_bull = hma_slow[i] > hma_slow[i-5]
            hma_slope_bear = hma_slow[i] < hma_slow[i-5]
        
        # === RSI PULLBACK ENTRY ===
        # Long: RSI pulled back to 35-50 in bullish trend
        # Short: RSI rallied to 50-65 in bearish trend
        rsi_pullback_long = 35.0 <= rsi[i] <= 55.0
        rsi_pullback_short = 45.0 <= rsi[i] <= 65.0
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        position_size = SIZE_BASE
        
        # Long entry: HMA bullish + RSI pullback
        if hma_bullish and rsi_pullback_long:
            # Increase size if HTF confirms
            if htf_bullish:
                position_size = SIZE_CONFIRMED
            desired_signal = position_size
        
        # Short entry: HMA bearish + RSI pullback
        elif hma_bearish and rsi_pullback_short:
            # Increase size if HTF confirms
            if htf_bearish:
                position_size = SIZE_CONFIRMED
            desired_signal = -position_size
        
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
        
        # === DISCRETIZE SIGNAL ===
        if desired_signal > 0:
            final_signal = SIZE_CONFIRMED if desired_signal >= SIZE_CONFIRMED * 0.9 else SIZE_BASE
        elif desired_signal < 0:
            final_signal = -SIZE_CONFIRMED if desired_signal <= -SIZE_CONFIRMED * 0.9 else -SIZE_BASE
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
                # Flip position
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
        
        signals[i] = final_signal
    
    return signals