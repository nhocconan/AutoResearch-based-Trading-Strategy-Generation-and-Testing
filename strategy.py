#!/usr/bin/env python3
"""
Experiment #242: 4h Primary + 1d/1w HTF — Simplified RSI Pullback + HMA Trend

Hypothesis: After 217 failed experiments, complexity is the enemy. This strategy 
uses PROVEN patterns from research notes:
1. RSI pullback entries (not extremes) in direction of HTF trend
2. HMA for faster trend response than EMA
3. ATR trailing stop for risk management
4. 1d HMA(50) for major trend bias, 1w HMA(50) for macro filter

Why this should work:
- RSI pullback (40/60 levels) generates MORE trades than extreme RSI (15/85)
- HTF trend filter prevents counter-trend disasters in bear markets
- 4h timeframe = 20-50 trades/year target (fee manageable)
- Discrete sizing (0.25/0.30) minimizes churn costs

Key difference from failed #238: Simpler entry logic, no complex regime switching.
Entry: RSI crosses 40 upward (long) or 60 downward (short) + HTF trend alignment.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_rsi_pullback_hma_1d1w_v1"
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1d HMA for major trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=50)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate and align 1w HMA for macro trend filter
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=50)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    hma_4h_fast = calculate_hma(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25  # 25% base position size
    SIZE_STRONG = 0.30  # 30% for strong signals
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    # Track RSI crosses
    prev_rsi = np.nan
    
    for i in range(250, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi[i] if not np.isnan(rsi[i]) else prev_rsi
            continue
        
        if np.isnan(hma_4h[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi[i] if not np.isnan(rsi[i]) else prev_rsi
            continue
        
        if np.isnan(hma_1d_aligned[i]) or np.isnan(hma_1w_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            prev_rsi = rsi[i] if not np.isnan(rsi[i]) else prev_rsi
            continue
        
        # === HTF BIAS (1d and 1w HMA) ===
        # Strong bull: price > 1d HMA AND 1d HMA > 1w HMA
        htf_strong_bull = (close[i] > hma_1d_aligned[i]) and (hma_1d_aligned[i] > hma_1w_aligned[i])
        # Strong bear: price < 1d HMA AND 1d HMA < 1w HMA
        htf_strong_bear = (close[i] < hma_1d_aligned[i]) and (hma_1d_aligned[i] < hma_1w_aligned[i])
        # Neutral bull: price > 1d HMA but 1w unclear
        htf_neutral_bull = close[i] > hma_1d_aligned[i]
        # Neutral bear: price < 1d HMA but 1w unclear
        htf_neutral_bear = close[i] < hma_1d_aligned[i]
        
        # === 4h LOCAL TREND ===
        hma_bull = close[i] > hma_4h[i]
        hma_bear = close[i] < hma_4h[i]
        hma_fast_bull = hma_4h_fast[i] > hma_4h[i] if not np.isnan(hma_4h_fast[i]) else False
        hma_fast_bear = hma_4h_fast[i] < hma_4h[i] if not np.isnan(hma_4h_fast[i]) else False
        
        # === SMA200 FILTER ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === RSI PULLBACK DETECTION ===
        # Long: RSI was <40, now crosses above 40 (pullback ending)
        rsi_cross_long = False
        if not np.isnan(prev_rsi) and not np.isnan(rsi[i]):
            rsi_cross_long = (prev_rsi < 40.0) and (rsi[i] >= 40.0)
        
        # Short: RSI was >60, now crosses below 60 (rally ending)
        rsi_cross_short = False
        if not np.isnan(prev_rsi) and not np.isnan(rsi[i]):
            rsi_cross_short = (prev_rsi > 60.0) and (rsi[i] <= 60.0)
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG ENTRY: RSI pullback + HTF bull + local trend confirmation
        if rsi_cross_long:
            if htf_strong_bull and hma_bull and above_sma200:
                desired_signal = SIZE_STRONG  # Strong signal
            elif htf_neutral_bull and hma_bull:
                desired_signal = SIZE_BASE  # Base signal
        
        # SHORT ENTRY: RSI pullback + HTF bear + local trend confirmation
        elif rsi_cross_short:
            if htf_strong_bear and hma_bear and below_sma200:
                desired_signal = -SIZE_STRONG  # Strong signal
            elif htf_neutral_bear and hma_bear:
                desired_signal = -SIZE_BASE  # Base signal
        
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
        prev_rsi = rsi[i]
    
    return signals