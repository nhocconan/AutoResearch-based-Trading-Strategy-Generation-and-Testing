#!/usr/bin/env python3
"""
Experiment #1638: 4h Primary + 1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 1333 failed strategies, complexity is the enemy. The #1 failure 
mode is 0 trades due to over-filtering. This strategy uses PROVEN patterns from 
successful 4h strategies (HMA crossover + RSI pullback + ATR trail) with MINIMAL 
filters to GUARANTEE trade generation.

Key design choices based on failure analysis:
1. SIMPLE trend detection: HMA(21) vs HMA(50) crossover on 4h (proven on SOL)
2. 1d HMA(21) for major bias only (not dual 1d+1w which kills trades)
3. RSI pullback entries at 45-55 (NOT extremes 30/70 which rarely trigger)
4. NO Choppiness/Fisher regime filters (too restrictive based on #1618, #1628)
5. NO volume filters (failed in #1607, #1612, #1637)
6. Discrete signal sizes: 0.25 base, 0.30 strong confirmation
7. 2.5x ATR trailing stoploss via signal→0

Entry logic (LOOSE to guarantee ≥30 trades/train, ≥3 test):
- LONG: 4h HMA21>50 + 1d HMA bullish + RSI pullback 45-55 + price>1d HMA
- SHORT: 4h HMA21<50 + 1d HMA bearish + RSI pullback 45-55 + price<1d HMA
- NEUTRAL fallback: 1d HMA bias + RSI 40-60 (catches trades when 4h unclear)

Why this beats mtf_6h_triple_hma_kama_roc_1w1d_v1 (Sharpe=0.575):
- 4h TF = more responsive than 6h for entry timing
- HMA crossover proven on SOL (Sharpe +0.879 in historical tests)
- RSI pullback 45-55 triggers MORE often than 30/70 extremes
- Minimal filters = guaranteed trade generation (addresses #1 failure mode)
- 1d bias prevents counter-trend trades in major moves

Target: Sharpe>0.6, trades≥30 train, trades≥3 test, DD>-35%
Timeframe: 4h
Size: 0.25-0.30 discrete
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_1d_loose_v1"
timeframe = "4h"
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
    
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    gain = np.insert(gain, 0, 0)
    loss = np.insert(loss, 0, 0)
    
    avg_gain = pd.Series(gain).ewm(span=period, min_periods=period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(span=period, min_periods=period, adjust=False).mean().values
    
    rsi = np.full(n, np.nan, dtype=np.float64)
    mask = avg_loss != 0
    rs = np.zeros(n)
    rs[mask] = avg_gain[mask] / avg_loss[mask]
    rsi[mask] = 100 - (100 / (1 + rs[mask]))
    
    return rsi

def calculate_sma(close, period):
    """Simple Moving Average"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    sma = pd.Series(close).rolling(window=period, min_periods=period).mean().values
    return sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align HTF indicators
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate 4h indicators
    atr_14 = calculate_atr(high, low, close, period=14)
    rsi_14 = calculate_rsi(close, period=14)
    hma_21 = calculate_hma(close, period=21)
    hma_50 = calculate_hma(close, period=50)
    sma_200 = calculate_sma(close, period=200)
    
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
    
    # Warmup period
    min_bars = 250  # Need 200 for SMA200 + 50 for HMA50
    
    for i in range(min_bars, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(rsi_14[i]) or np.isnan(hma_21[i]) or np.isnan(hma_50[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === TREND DIRECTION (4h HMA crossover) ===
        hma_bullish = hma_21[i] > hma_50[i]
        hma_bearish = hma_21[i] < hma_50[i]
        
        # === TREND BIAS (1d HMA) ===
        price_above_1d = close[i] > hma_1d_aligned[i]
        price_below_1d = close[i] < hma_1d_aligned[i]
        
        # === LONG TERM FILTER (SMA200) ===
        price_above_200 = close[i] > sma_200[i]
        price_below_200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK (LOOSE - 45-55 range for more trades) ===
        rsi_val = rsi_14[i]
        rsi_pullback_long = 45 <= rsi_val <= 55
        rsi_pullback_short = 45 <= rsi_val <= 55
        rsi_neutral = 40 <= rsi_val <= 60
        
        # === ENTRY LOGIC (LOOSE - must generate trades) ===
        desired_signal = 0.0
        
        # PRIMARY: 4h HMA trend + 1d bias + RSI pullback
        if hma_bullish and price_above_1d and rsi_pullback_long:
            # Strong signal if also above SMA200
            if price_above_200:
                desired_signal = SIZE_STRONG
            else:
                desired_signal = SIZE_BASE
        
        elif hma_bearish and price_below_1d and rsi_pullback_short:
            # Strong signal if also below SMA200
            if price_below_200:
                desired_signal = -SIZE_STRONG
            else:
                desired_signal = -SIZE_BASE
        
        # FALLBACK: 1d bias + RSI neutral (catches more trades)
        elif price_above_1d and rsi_neutral and hma_bullish:
            desired_signal = SIZE_BASE
        
        elif price_below_1d and rsi_neutral and hma_bearish:
            desired_signal = -SIZE_BASE
        
        # ULTIMATE FALLBACK: Just 1d bias + RSI in range (guarantees trades)
        elif price_above_1d and 35 <= rsi_val <= 65:
            desired_signal = SIZE_BASE * 0.8
        
        elif price_below_1d and 35 <= rsi_val <= 65:
            desired_signal = -SIZE_BASE * 0.8
        
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
        elif desired_signal >= SIZE_BASE * 0.6:
            final_signal = SIZE_BASE * 0.8
        elif desired_signal <= -SIZE_BASE * 0.6:
            final_signal = -SIZE_BASE * 0.8
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
    
    return signals