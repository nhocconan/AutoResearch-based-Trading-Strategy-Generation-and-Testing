#!/usr/bin/env python3
"""
Experiment #1429: 4h Primary + 1d HTF — HMA Crossover + RSI Filter + ATR Stop

Hypothesis: 4h timeframe with 1d HMA trend filter will generate consistent trades
while maintaining positive Sharpe. Based on research patterns:
1. HMA crossover + RSI filter worked on SOL (Sharpe +0.879)
2. Donchian breakout + HMA trend + RSI worked on SOL (Sharpe +0.782)
3. 4h needs simpler logic than 1d to avoid 0-trade failure mode

Why 4h not 1d:
- 4h generates more trades than 1d (target 20-50/year vs 10-30/year)
- Still filters noise better than 1h/30m
- Proven to work with HTF trend filter

Design:
1. 1d HMA(21) = macro trend direction (call ONCE before loop)
2. 4h HMA(16) vs HMA(48) crossover = entry trigger
3. RSI(14) filter: avoid entries when RSI > 70 (long) or RSI < 30 (short)
4. ATR(14) trailing stop 2.5x = risk management
5. Position size 0.30 = conservative for 4h volatility

Key: Entry conditions RELAXED to ensure >= 10 trades train, >= 3 test
- HMA crossover alone (no ADX, no Choppiness, no volume filters)
- RSI filter only blocks extreme entries, doesn't require extremes
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_crossover_rsi_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - faster response than EMA, less lag"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    def wma(series, span):
        weights = np.arange(1, span + 1, dtype=np.float64)
        result = np.full(len(series), np.nan)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            if not np.any(np.isnan(window)):
                result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    hma = np.full(n, np.nan)
    for i in range(period - 1, n):
        if not np.isnan(wma_half[i]) and not np.isnan(wma_full[i]):
            diff = 2.0 * wma_half[i] - wma_full[i]
            if i >= sqrt_period - 1:
                diff_vals = []
                for j in range(i - sqrt_period + 1, i + 1):
                    if j >= period - 1 and not np.isnan(2.0 * wma_half[j] - wma_full[j]):
                        diff_vals.append(2.0 * wma_half[j] - wma_full[j])
                if len(diff_vals) == sqrt_period:
                    weights = np.arange(1, sqrt_period + 1, dtype=np.float64)
                    hma[i] = np.sum(np.array(diff_vals) * weights) / np.sum(weights)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index"""
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
    """Average True Range - for stoploss sizing"""
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
    
    # Calculate and align 1d HMA for macro trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    hma_16 = calculate_hma(close, period=16)
    hma_48 = calculate_hma(close, period=48)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    
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
        if np.isnan(hma_16[i]) or np.isnan(hma_48[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(rsi[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === MACRO TREND (1d HMA) ===
        macro_bull = close[i] > hma_1d_aligned[i]
        macro_bear = close[i] < hma_1d_aligned[i]
        
        # === HMA CROSSOVER (4h) ===
        hma_bull_cross = hma_16[i] > hma_48[i] and hma_16[i-1] <= hma_48[i-1]
        hma_bear_cross = hma_16[i] < hma_48[i] and hma_16[i-1] >= hma_48[i-1]
        
        # === HMA ALIGNMENT (already aligned, not crossover) ===
        hma_bull_aligned = hma_16[i] > hma_48[i]
        hma_bear_aligned = hma_16[i] < hma_48[i]
        
        # === RSI FILTER ===
        rsi_not_overbought = rsi[i] < 70.0
        rsi_not_oversold = rsi[i] > 30.0
        rsi_bullish = rsi[i] > 50.0
        rsi_bearish = rsi[i] < 50.0
        
        # === DESIRED SIGNAL - SIMPLIFIED LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRIES
        # Path 1: HMA bull crossover + macro bull + RSI filter
        if hma_bull_cross and macro_bull and rsi_not_overbought:
            desired_signal = BASE_SIZE
        # Path 2: HMA already bull aligned + macro bull + RSI bullish (continuation)
        elif hma_bull_aligned and macro_bull and rsi_bullish and rsi_not_overbought:
            # Check if we just crossed above 1d HMA
            if close[i] > hma_1d_aligned[i] and close[i-1] <= hma_1d_aligned[i-1]:
                desired_signal = BASE_SIZE
        
        # SHORT ENTRIES
        # Path 1: HMA bear crossover + macro bear + RSI filter
        elif hma_bear_cross and macro_bear and rsi_not_oversold:
            desired_signal = -BASE_SIZE
        # Path 2: HMA already bear aligned + macro bear + RSI bearish (continuation)
        elif hma_bear_aligned and macro_bear and rsi_bearish and rsi_not_oversold:
            # Check if we just crossed below 1d HMA
            if close[i] < hma_1d_aligned[i] and close[i-1] >= hma_1d_aligned[i-1]:
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
        if desired_signal >= BASE_SIZE * 0.5:
            final_signal = BASE_SIZE if desired_signal > 0 else -BASE_SIZE
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