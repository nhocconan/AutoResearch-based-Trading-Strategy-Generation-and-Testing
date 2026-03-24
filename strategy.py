#!/usr/bin/env python3
"""
Experiment #100: 1h Primary + 4h/12h HTF — Triple HMA Trend + RSI Entry

Hypothesis: After 99 experiments, the clearest lesson is:
- Too many filters = 0 trades (experiments #088-#099 all failed with Sharpe=0.000 or negative)
- 1h timeframe needs HTF (4h/12h) for direction, 1h only for entry timing
- RSI thresholds must be LOOSE (40-60, not 30-70) to ensure trade generation
- Position size should be smaller for 1h (0.25 vs 0.30-0.35 for 4h/12h)

This strategy uses 3 confluence filters (minimum for quality, maximum for trade gen):
1. 12h HMA(50) = major trend bias (price above/below)
2. 4h HMA(16/48) crossover = intermediate trend confirmation
3. 1h RSI(14) = entry timing (loose: >40 for long, <60 for short)
4. ATR(14) trailing stoploss = 2.5x for risk management

Key design choices:
- Timeframe: 1h (target 40-80 trades/year)
- HTF: 4h for intermediate trend, 12h for major bias
- RSI thresholds: 40/60 (loose, ensures trades on all symbols)
- Position size: 0.25 (conservative for 1h, reduces fee drag)
- Stoploss: 2.5x ATR trailing (proven in previous experiments)

Why this should work:
- 12h HMA ensures we trade with major trend (avoid counter-trend losses)
- 4h HMA crossover confirms intermediate momentum (reduces whipsaws)
- Loose RSI ensures we get entries on pullbacks within the trend
- 0.25 size means -77% BTC crash = -19% equity (survivable)

Target: Sharpe>0.351, DD>-40%, trades>=30 on train, trades>=3 on test
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_triple_hma_rsi_4h12h_v1"
timeframe = "1h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average - smoother and more responsive than EMA"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    close_series = pd.Series(close)
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = close_series.ewm(span=period // 2, min_periods=period // 2, adjust=False).mean()
    wma_full = close_series.ewm(span=period, min_periods=period, adjust=False).mean()
    
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=int(np.sqrt(period)), min_periods=int(np.sqrt(period)), adjust=False).mean()
    
    return hma.values

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
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for major trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=50)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate and align 4h HMA for intermediate trend
    hma_4h_fast_raw = calculate_hma(df_4h['close'].values, period=16)
    hma_4h_slow_raw = calculate_hma(df_4h['close'].values, period=48)
    hma_4h_fast_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_fast_raw)
    hma_4h_slow_aligned = align_htf_to_ltf(prices, df_4h, hma_4h_slow_raw)
    
    # Calculate primary (1h) indicators
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (conservative for 1h)
    
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
        if np.isnan(hma_12h_aligned[i]) or np.isnan(hma_4h_fast_aligned[i]) or np.isnan(hma_4h_slow_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS (12h HMA) ===
        # Is price above or below 12h HMA(50)?
        htf_bull = close[i] > hma_12h_aligned[i]
        htf_bear = close[i] < hma_12h_aligned[i]
        
        # === INTERMEDIATE TREND (4h HMA crossover) ===
        hma_4h_cross_bull = hma_4h_fast_aligned[i] > hma_4h_slow_aligned[i]
        hma_4h_cross_bear = hma_4h_fast_aligned[i] < hma_4h_slow_aligned[i]
        
        # === RSI FILTER (LOOSE - ensure trades generate) ===
        # For longs: RSI > 40 (not deeply oversold, but not overbought)
        # For shorts: RSI < 60 (not deeply overbought, but not oversold)
        rsi_ok_long = rsi[i] > 40.0
        rsi_ok_short = rsi[i] < 60.0
        
        # === DESIRED SIGNAL ===
        # LONG: 12h bull + 4h HMA cross bull + RSI > 40
        # SHORT: 12h bear + 4h HMA cross bear + RSI < 60
        desired_signal = 0.0
        
        if htf_bull and hma_4h_cross_bull and rsi_ok_long:
            desired_signal = SIZE
        elif htf_bear and hma_4h_cross_bear and rsi_ok_short:
            desired_signal = -SIZE
        
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