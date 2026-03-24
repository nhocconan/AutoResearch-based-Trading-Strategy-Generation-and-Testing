#!/usr/bin/env python3
"""
Experiment #1540: 1h Primary + 4h/12h HTF — Simplified Trend-Pullback Strategy

Hypothesis: After analyzing failures #1528-1539, the critical pattern is:
1. 1h strategies with strict filters (session, volume, chop) → 0 trades (#1530, #1535, #1538)
2. Complex regime detection (Choppiness + Connors RSI) → negative Sharpe (#1534)
3. LOOSE entry conditions are MANDATORY for lower TFs to generate trades

New approach for 1h (learned from failures):
- 12h HMA(21): Macro trend bias (only trade in direction)
- 4h HMA(21): Intermediate trend confirmation
- 1h RSI(14): Pullback entry (RSI<40 long, RSI>60 short — LOOSE thresholds!)
- 1h HMA(21/55): Additional trend confirmation
- NO volume filter (caused 0 trades in #1538)
- NO session filter (caused 0 trades in #1530)
- ATR(14) 2.5x trailing stop
- Position size: 0.25 (conservative for 1h volatility)

Why this should work:
- LOOSE RSI thresholds (40/60 not 30/70) → ensures trades fire
- HTF bias prevents counter-trend trades in 2022 crash
- NO strict volume/session filters → avoids 0-trade problem
- Discrete sizing (0.0, ±0.25) minimizes fee churn
- Target: 50-100 trades/train, 10-20 trades/test

Timeframe: 1h (required by experiment)
HTF: 4h (intermediate), 12h (macro)
Position Size: 0.25
Target: Sharpe > 0.618, DD < -40%, trades > 30
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_hma_rsi_pullback_4h12h_atr_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period=21):
    """
    Hull Moving Average - reduces lag while maintaining smoothness
    HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    """
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(data, w_period):
        result = np.full(len(data), np.nan)
        if w_period < 1:
            return result
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
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 4h HMA
    hma_4h_raw = calculate_hma(df_4h['close'].values, period=21)
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_raw)
    
    # Calculate and align 12h HMA
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate 1h indicators
    hma_21 = calculate_hma(close, period=21)
    hma_55 = calculate_hma(close, period=55)
    rsi_14 = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.25
    
    # Position tracking
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
        if np.isnan(rsi_14[i]) or np.isnan(hma_21[i]) or np.isnan(hma_55[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_4h_aligned[i]) or np.isnan(hma_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (12h HMA) ===
        twelveh_bull = close[i] > hma_12h_aligned[i]
        twelveh_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA) ===
        fourh_bull = close[i] > hma_4h_aligned[i]
        fourh_bear = close[i] < hma_4h_aligned[i]
        
        # === PRIMARY TREND (1h HMA Crossover) ===
        hma_fast_above_slow = hma_21[i] > hma_55[i]
        hma_fast_below_slow = hma_21[i] < hma_55[i]
        
        # === RSI PULLBACK (LOOSE thresholds for trade frequency) ===
        rsi_oversold = rsi_14[i] < 40.0  # Loose: was 30 in failed strategies
        rsi_overbought = rsi_14[i] > 60.0  # Loose: was 70 in failed strategies
        
        # === DESIRED SIGNAL — 3 CONFLUENCE MINIMUM ===
        desired_signal = 0.0
        
        # Long: 12h bull + (4h bull OR 1h HMA cross) + RSI pullback
        if twelveh_bull:
            if (fourh_bull or hma_fast_above_slow) and rsi_oversold:
                desired_signal = BASE_SIZE
        
        # Short: 12h bear + (4h bear OR 1h HMA cross) + RSI pullback
        elif twelveh_bear:
            if (fourh_bear or hma_fast_below_slow) and rsi_overbought:
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
        if desired_signal >= BASE_SIZE * 0.8:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.8:
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
        
        signals[i] = final_signal
    
    return signals