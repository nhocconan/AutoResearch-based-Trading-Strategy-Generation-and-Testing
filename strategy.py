#!/usr/bin/env python3
"""
Experiment #054: 4h Primary + 12h/1d HTF — Simplified HMA Trend + RSI Pullback

Hypothesis: After 50+ failed experiments, the #1 issue is TOO STRICT entry conditions
resulting in 0 trades. This strategy SIMPLIFIES entry logic to ensure trade generation:

1. 4h HMA(21) for primary trend direction (proven in best strategy)
2. 12h HMA(21) for major trend confirmation (looser than 1d/1w)
3. RSI(14) pullback entries - LESS extreme thresholds (40/60 vs 15/85)
4. Simple ATR trailing stop (2.5x)
5. Discrete signal sizes: 0.0, ±0.25, ±0.30

Key changes from failed experiments:
- RSI thresholds 40/60 (vs CRSI 15/85) - MUCH more trade opportunities
- Only 2 HTF levels (12h) instead of 3 (1d+1w) - less conflicting signals
- Removed Choppiness Index regime - was causing 0 trades in neutral zones
- Removed funding rate - loading issues causing 0 trades
- Simpler logic = more trades = actual Sharpe calculation

Entry Logic:
- LONG: price > 4h HMA + price > 12h HMA + RSI(14) < 45 (pullback in uptrend)
- SHORT: price < 4h HMA + price < 12h HMA + RSI(14) > 55 (rally in downtrend)
- Size: 0.30 with full HTF alignment, 0.25 with partial

Risk: 2.5x ATR trailing stop, max signal 0.35, discrete levels
Target: Sharpe>0.4, trades>40/symbol train (loose entries), >5/symbol test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_hma_rsi_pullback_12h_simplified_v1"
timeframe = "4h"
leverage = 1.0

def calculate_hma(close, period=21):
    """Hull Moving Average - responsive trend indicator"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half_period = max(period // 2, 1)
    sqrt_period = max(int(np.sqrt(period)), 1)
    
    def wma(series, span):
        if len(series) < span:
            return np.full(len(series), np.nan)
        result = np.full(len(series), np.nan)
        weights = np.arange(1, span + 1, dtype=float)
        for i in range(span - 1, len(series)):
            window = series[i - span + 1:i + 1]
            result[i] = np.sum(window * weights) / np.sum(weights)
        return result
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    raw_hma = 2.0 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_rsi(close, period=14):
    """Relative Strength Index - momentum oscillator"""
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
    
    rsi = np.full(n, np.nan)
    for i in range(period, n):
        if avg_loss[i] < 1e-10:
            rsi[i] = 100.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi

def calculate_atr(high, low, close, period=14):
    """Average True Range - volatility measure for stops"""
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
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h HMA for major trend bias
    hma_12h_raw = calculate_hma(df_12h['close'].values, period=21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_raw)
    
    # Calculate primary (4h) indicators
    hma_4h = calculate_hma(close, period=21)
    rsi = calculate_rsi(close, period=14)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    FULL_SIZE = 0.30
    REDUCED_SIZE = 0.25
    MAX_SIZE = 0.35
    
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
        if np.isnan(hma_4h[i]) or np.isnan(hma_12h_aligned[i]):
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
        
        # === TREND DIRECTION ===
        price_above_hma_4h = close[i] > hma_4h[i]
        price_above_hma_12h = close[i] > hma_12h_aligned[i]
        price_below_hma_4h = close[i] < hma_4h[i]
        price_below_hma_12h = close[i] < hma_12h_aligned[i]
        
        # === RSI PULLBACK SIGNALS (LOOSE THRESHOLDS FOR TRADE GEN) ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # LONG: Uptrend + RSI pullback (threshold 45 - much looser than 15)
        if price_above_hma_4h and price_above_hma_12h:
            if rsi[i] < 45.0:  # Pullback in uptrend
                signal_strength = FULL_SIZE
                desired_signal = signal_strength
            elif rsi[i] < 50.0:  # Weaker pullback
                signal_strength = REDUCED_SIZE
                desired_signal = signal_strength
        
        # SHORT: Downtrend + RSI rally (threshold 55 - much looser than 85)
        elif price_below_hma_4h and price_below_hma_12h:
            if rsi[i] > 55.0:  # Rally in downtrend
                signal_strength = FULL_SIZE
                desired_signal = -signal_strength
            elif rsi[i] > 50.0:  # Weaker rally
                signal_strength = REDUCED_SIZE
                desired_signal = -signal_strength
        
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
        desired_signal = np.clip(desired_signal, -MAX_SIZE, MAX_SIZE)
        
        if desired_signal >= FULL_SIZE * 0.85:
            final_signal = FULL_SIZE
        elif desired_signal <= -FULL_SIZE * 0.85:
            final_signal = -FULL_SIZE
        elif desired_signal >= REDUCED_SIZE * 0.85:
            final_signal = REDUCED_SIZE
        elif desired_signal <= -REDUCED_SIZE * 0.85:
            final_signal = -REDUCED_SIZE
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