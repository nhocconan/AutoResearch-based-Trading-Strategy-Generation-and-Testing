#!/usr/bin/env python3
"""
Experiment #1203: 6h Primary + 1d/1w HTF — Dual HMA Cross with RSI Momentum

Hypothesis: After 990+ failed experiments, the key insight is:
1. Donchian breakouts on 6h are TOO RARE (2-3/year) → 0 trades with filters
2. Simple HMA trend + RSI pullback WORKS (exp 1198 Sharpe=0.141, current best Sharpe=0.445)
3. DUAL HMA cross on daily provides cleaner trend signal than price-vs-HMA
4. RSI momentum (not pullback) generates MORE trades while staying trend-aligned

Strategy logic:
- LONG: Daily HMA(9) > HMA(21) [bullish] + 6h RSI(14) > 50 [momentum] + Weekly HMA up
- SHORT: Daily HMA(9) < HMA(21) [bearish] + 6h RSI(14) < 50 [momentum] + Weekly HMA down
- Exit: RSI crosses below 50 (long) or above 50 (short) OR stoploss hit

Why this generates trades:
- HMA cross changes ~10-20 times/year on daily
- RSI >50/<50 flips frequently within trend
- Combined = 30-60 trades/year target
- No choppiness/ADX filters that kill signal generation

Target: Sharpe>0.5, trades>=30 train, trades>=5 test, DD>-35%
Timeframe: 6h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_dual_hma_cross_rsi_momentum_1d1w_v1"
timeframe = "6h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - reduces lag while smoothing"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = max(1, period // 2)
    sqrt_n = max(1, int(np.sqrt(period)))
    
    def wma(series, span):
        result = np.full(len(series), np.nan, dtype=np.float64)
        weights = np.arange(1, span + 1, dtype=np.float64)
        weight_sum = np.sum(weights)
        for i in range(span - 1, len(series)):
            if not np.isnan(series[i]):
                window = series[i - span + 1:i + 1].astype(np.float64)
                if not np.any(np.isnan(window)):
                    result[i] = np.sum(window * weights) / weight_sum
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    diff = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff[i] = 2.0 * wma_half[i] - wma_full[i]
    
    return wma(diff, sqrt_n)

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss != 0)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi[:period] = np.nan
    return rsi

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF indicators
    hma_1d_fast_raw = calculate_hma(df_1d['close'].values, period=9)
    hma_1d_slow_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_fast = align_htf_to_ltf(prices, df_1d, hma_1d_fast_raw)
    hma_1d_slow = align_htf_to_ltf(prices, df_1d, hma_1d_slow_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate 6h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    
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
    
    # Track previous RSI for exit detection
    prev_rsi = 50.0
    
    # Warmup period
    min_bars = 100
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi_14[i] if not np.isnan(rsi_14[i]) else prev_rsi
            continue
        
        if np.isnan(rsi_14[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_fast[i]) or np.isnan(hma_1d_slow[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi_14[i] if not np.isnan(rsi_14[i]) else prev_rsi
            continue
        
        # === TREND DIRECTION (Daily HMA Cross) ===
        hma_cross_bullish = hma_1d_fast[i] > hma_1d_slow[i]
        hma_cross_bearish = hma_1d_fast[i] < hma_1d_slow[i]
        
        # Weekly HMA slope for major trend confirmation
        hma_1w_valid = not np.isnan(hma_1w_aligned[i])
        hma_1w_up = False
        hma_1w_down = False
        if hma_1w_valid and i >= 5 and not np.isnan(hma_1w_aligned[i-5]):
            hma_1w_up = hma_1w_aligned[i] > hma_1w_aligned[i-5]
            hma_1w_down = hma_1w_aligned[i] < hma_1w_aligned[i-5]
        
        # === MOMENTUM (6h RSI) ===
        rsi = rsi_14[i]
        rsi_momentum_long = rsi > 50.0
        rsi_momentum_short = rsi < 50.0
        
        # RSI cross detection for exit
        rsi_crossed_below_50 = (prev_rsi >= 50.0) and (rsi < 50.0)
        rsi_crossed_above_50 = (prev_rsi <= 50.0) and (rsi > 50.0)
        
        # === ENTRY LOGIC (LOOSE - guarantee trades) ===
        desired_signal = 0.0
        
        # LONG: Daily HMA cross bullish + RSI momentum > 50
        if hma_cross_bullish and rsi_momentum_long:
            if hma_1w_up:
                desired_signal = SIZE_STRONG  # Strong trend alignment
            else:
                desired_signal = SIZE_BASE  # Basic bullish cross
        
        # SHORT: Daily HMA cross bearish + RSI momentum < 50
        elif hma_cross_bearish and rsi_momentum_short:
            if hma_1w_down:
                desired_signal = -SIZE_STRONG  # Strong trend alignment
            else:
                desired_signal = -SIZE_BASE  # Basic bearish cross
        
        # === EXIT LOGIC (RSI cross against position) ===
        if in_position and position_side > 0 and rsi_crossed_below_50:
            desired_signal = 0.0  # Exit long on RSI cross below 50
        
        if in_position and position_side < 0 and rsi_crossed_above_50:
            desired_signal = 0.0  # Exit short on RSI cross above 50
        
        # === STOPLOSS CHECK (2.5x ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
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
        else:
            final_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if final_signal != 0.0:
            if not in_position or np.sign(final_signal) != position_side:
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
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
        prev_rsi = rsi
    
    return signals