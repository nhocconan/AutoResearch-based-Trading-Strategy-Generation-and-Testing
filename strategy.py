#!/usr/bin/env python3
"""
Experiment #346: 1d Primary + 1w HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: Previous failures due to overly complex regime filters and too many
confluence requirements causing 0 trades. Return to proven pattern:
- 1w HMA for major trend bias (only trade in direction of weekly trend)
- 1d HMA(21/50) crossover for trend direction
- RSI(14) pullback entry (not extreme, just 40-60 range in trend)
- ATR stoploss 2.5x from entry
- Discrete sizing: 0.25 base, 0.30 when 1w aligned

Key changes from failed experiments:
1. REMOVED Choppiness Index (caused regime flip-flop and 0 trades)
2. LOOSENED RSI entry: 35-65 range instead of 30/70 extremes
3. SIMPLIFIED entry: HMA crossover + RSI confirmation + 1w bias (3 filters max)
4. ENSURE trades: entry triggers on any HMA cross in 1w trend direction

Target: 25-40 trades/year on 1d, Sharpe>0.40, DD>-35%, ALL symbols positive
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_hma_rsi_weekly_bias_v1"
timeframe = "1d"
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
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align 1w HMA for major trend bias
    hma_1w_raw = calculate_hma(df_1w['close'].values, period=21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (1d) indicators
    hma_1d_fast = calculate_hma(close, period=21)
    hma_1d_slow = calculate_hma(close, period=50)
    atr = calculate_atr(high, low, close, period=14)
    rsi = calculate_rsi(close, period=14)
    sma_200 = calculate_sma(close, 200)
    
    signals = np.zeros(n)
    SIZE_BASE = 0.25
    SIZE_STRONG = 0.30
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    
    for i in range(250, n):
        # Skip if indicators not ready
        if np.isnan(atr[i]) or atr[i] <= 1e-10:
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
            continue
        
        if np.isnan(rsi[i]) or np.isnan(sma_200[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === 1W HTF BIAS (major trend direction) ===
        hma_1w_valid = not np.isnan(hma_1w_aligned[i])
        htf_bull = hma_1w_valid and close[i] > hma_1w_aligned[i]
        htf_bear = hma_1w_valid and close[i] < hma_1w_aligned[i]
        
        # === 1D HMA TREND (fast vs slow) ===
        hma_bull = hma_1d_fast[i] > hma_1d_slow[i]
        hma_bear = hma_1d_fast[i] < hma_1d_slow[i]
        
        # === HMA CROSSOVER DETECTION ===
        hma_cross_long = False
        hma_cross_short = False
        if i > 0 and not np.isnan(hma_1d_fast[i-1]) and not np.isnan(hma_1d_slow[i-1]):
            # Fast crosses above slow
            if hma_1d_fast[i-1] <= hma_1d_slow[i-1] and hma_1d_fast[i] > hma_1d_slow[i]:
                hma_cross_long = True
            # Fast crosses below slow
            if hma_1d_fast[i-1] >= hma_1d_slow[i-1] and hma_1d_fast[i] < hma_1d_slow[i]:
                hma_cross_short = True
        
        # === RSI PULLBACK (LOOSENED for more trades) ===
        # In uptrend: enter on RSI pullback to 40-55 range
        rsi_pullback_long = 35.0 < rsi[i] < 60.0
        # In downtrend: enter on RSI bounce to 40-65 range
        rsi_pullback_short = 40.0 < rsi[i] < 65.0
        
        # === SMA200 FILTER (secular trend) ===
        above_sma200 = close[i] > sma_200[i]
        below_sma200 = close[i] < sma_200[i]
        
        # === ENTRY LOGIC (SIMPLIFIED - 3 filters max) ===
        desired_signal = 0.0
        
        # LONG: 1w bull + 1d HMA bull + RSI pullback OR HMA cross
        if htf_bull and hma_bull:
            if hma_cross_long:
                # Crossover entry - strongest signal
                desired_signal = SIZE_STRONG
            elif rsi_pullback_long and above_sma200:
                # Pullback entry in established trend
                desired_signal = SIZE_BASE
        
        # SHORT: 1w bear + 1d HMA bear + RSI pullback OR HMA cross
        elif htf_bear and hma_bear:
            if hma_cross_short:
                # Crossover entry - strongest signal
                desired_signal = -SIZE_STRONG
            elif rsi_pullback_short and below_sma200:
                # Pullback entry in established trend
                desired_signal = -SIZE_BASE
        
        # === STOPLOSS CHECK (2.5x ATR from entry) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
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
                # New position or flip
                in_position = True
                position_side = int(np.sign(final_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                # Set stoploss
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
        
        signals[i] = final_signal
    
    return signals