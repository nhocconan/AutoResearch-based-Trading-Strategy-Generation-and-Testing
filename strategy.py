#!/usr/bin/env python3
"""
Experiment #023: 6h Primary + 1d/1w HTF — Dual HMA Trend + RSI Pullback

Hypothesis: After 22 failed experiments, the pattern is clear:
- Complex regime switching (Choppiness + ADX + multiple filters) = too restrictive = 0 trades
- 6h timeframe needs SIMPLER logic than 4h/12h
- SOLUTION: Dual HMA trend (1w + 1d) for direction, 6h RSI pullback for entry
- This is proven pattern from traditional trend-following: trade WITH higher TF trend,
  enter on lower TF pullbacks (RSI oversold in uptrend, overbought in downtrend)
- 1w HMA(50) = major trend (very slow, reduces whipsaw)
- 1d HMA(21) = intermediate trend confirmation
- 6h RSI(14) pullback = entry timing (buy dips in uptrend, sell rallies in downtrend)
- LOOSE RSI thresholds (35/65) ensure enough trades on all symbols
- ATR stoploss for risk management

Key design choices:
- Timeframe: 6h (30-60 trades/year target)
- HTF: 1w HMA(50) + 1d HMA(21) for trend bias
- Entry: RSI pullback in trend direction (NOT breakout = fewer false signals)
- Position size: 0.28 (28% of capital)
- Stoploss: 2.5x ATR trailing
- LOOSE filters: RSI<50 for long in uptrend, RSI>50 for short in downtrend

Why this should work on 6h:
- 6h is middle ground: slower than 4h (less noise), faster than 12h (more trades)
- Dual HMA filter prevents counter-trend trades (major failure in 2022 crash)
- RSI pullback entries = better risk/reward than breakouts (enter at support/resistance)
- Simpler logic = more trades = meets minimum trade requirement

Target: Sharpe>0.019 (beat current best), DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_dual_hma_rsi_pullback_1d1w_v1"
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
    """Average True Range for stoploss"""
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA(50) for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate and align 1d HMA(21) for intermediate trend
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (6h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # 28% position size
    
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
        if np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(hma_1w_aligned[i]) or np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (Dual HMA) ===
        # Major trend: price vs 1w HMA(50)
        htf_major_bull = close[i] > hma_1w_aligned[i]
        htf_major_bear = close[i] < hma_1w_aligned[i]
        
        # Intermediate trend: price vs 1d HMA(21)
        htf_inter_bull = close[i] > hma_1d_aligned[i]
        htf_inter_bear = close[i] < hma_1d_aligned[i]
        
        # === RSI PULLBACK SIGNALS ===
        # Long: uptrend + RSI pullback (oversold in uptrend = buying opportunity)
        # Short: downtrend + RSI pullback (overbought in downtrend = selling opportunity)
        
        # LOOSE thresholds to ensure trades: RSI < 50 for long, RSI > 50 for short
        rsi_pullback_long = rsi[i] < 50.0
        rsi_pullback_short = rsi[i] > 50.0
        
        # Stronger signal: RSI < 40 (deep pullback) or RSI > 60 (strong rally in downtrend)
        rsi_deep_long = rsi[i] < 40.0
        rsi_deep_short = rsi[i] > 60.0
        
        # === DESIRED SIGNAL (Trend + Pullback) ===
        desired_signal = 0.0
        
        # LONG: Major uptrend + intermediate uptrend + RSI pullback
        if htf_major_bull and htf_inter_bull and rsi_pullback_long:
            if rsi_deep_long:
                desired_signal = SIZE  # Strong signal
            else:
                desired_signal = SIZE * 0.7  # Moderate signal
        
        # SHORT: Major downtrend + intermediate downtrend + RSI pullback
        elif htf_major_bear and htf_inter_bear and rsi_pullback_short:
            if rsi_deep_short:
                desired_signal = -SIZE  # Strong signal
            else:
                desired_signal = -SIZE * 0.7  # Moderate signal
        
        # Fallback: Only major trend + RSI (ignore intermediate if strong pullback)
        elif htf_major_bull and rsi_deep_long:
            desired_signal = SIZE * 0.5
        elif htf_major_bear and rsi_deep_short:
            desired_signal = -SIZE * 0.5
        
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
        if desired_signal >= SIZE * 0.85:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.85:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.5:
            final_signal = SIZE * 0.5
        elif desired_signal <= -SIZE * 0.5:
            final_signal = -SIZE * 0.5
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