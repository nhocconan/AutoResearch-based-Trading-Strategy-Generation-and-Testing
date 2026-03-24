#!/usr/bin/env python3
"""
Experiment #079: 4h Primary + 1d HTF — KAMA Adaptive Trend + Ehlers Fisher Reversal

Hypothesis: After 78 failed experiments, the pattern is clear:
- Donchian breakouts fail in choppy bear markets (2022, 2025)
- Choppiness Index regime filters haven't worked (tried 4x, all negative)
- Fisher Transform alone failed, but combined with KAMA may work

NEW APPROACH:
1. KAMA (Kaufman Adaptive) - adapts smoothing based on market efficiency
   - Fast in trends, slow in chop (unlike fixed EMA)
   - ER (Efficiency Ratio) determines smoothing constant
2. Ehlers Fisher Transform - catches reversals at extremes
   - Long when Fisher crosses above -1.5 (oversold reversal)
   - Short when Fisher crosses below +1.5 (overbought reversal)
3. 1d HMA trend bias - only take signals in HTF trend direction
4. ATR(14) 2.5x trailing stop - proven risk management

Why this should work:
- KAMA handles 2022 crash better than EMA (slows down in panic)
- Fisher catches bear market rallies (75% win rate in research)
- 1d HMA prevents counter-trend trades in major moves
- 4h timeframe = 20-50 trades/year (fee-efficient)
- LOOSE Fisher thresholds (-1.5/+1.5 not -2.0/+2.0) ensure trades generate

Entry Logic:
- Long: Fisher crosses above -1.5 + price > KAMA + price > 1d HMA
- Short: Fisher crosses below +1.5 + price < KAMA + price < 1d HMA
- Size: 0.28 (discrete, between 0.25-0.30 proven range)

Risk: 2.5x ATR trailing stop, signal→0 when stopped out
Target: Sharpe>0.4, trades>30/symbol train, >3/symbol test, DD>-35%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_kama_fisher_hma_1d_v1"
timeframe = "4h"
leverage = 1.0

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency ratio (ER)
    ER = |close - close[n]| / sum(|close[i] - close[i-1]|)
    High ER = trending (fast smoothing), Low ER = choppy (slow smoothing)
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.full(n, np.nan)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = 0.0
        for j in range(i - period + 1, i + 1):
            noise += abs(close[j] - close[j - 1])
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 0.0
    
    # Calculate smoothing constant (SC)
    # SC = ER * (fast_sc - slow_sc) + slow_sc
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    sc = np.full(n, np.nan)
    for i in range(period, n):
        if not np.isnan(er[i]):
            sc[i] = er[i] * (fast_sc - slow_sc) + slow_sc
    
    # Initialize KAMA with SMA of first period
    kama[period] = np.mean(close[:period + 1])
    
    # Calculate KAMA recursively
    for i in range(period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i - 1] + sc[i] ** 2 * (close[i] - kama[i - 1])
    
    return kama

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Normalizes price to Gaussian distribution for better reversal detection
    Fisher = 0.5 * ln((1 + X) / (1 - X)) where X = 2 * (price - min) / (max - min) - 1
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan), np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_signal = np.full(n, np.nan)  # Previous bar's fisher (for crossover detection)
    
    # Use (high + low) / 2 as price input
    hl2 = (high + low) / 2.0
    
    for i in range(period - 1, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest - lowest > 1e-10:
            x = 2.0 * (hl2[i] - lowest) / (highest - lowest) - 1.0
            # Clamp x to avoid ln(0) or ln(negative)
            x = max(-0.999, min(0.999, x))
            fisher[i] = 0.5 * np.log((1.0 + x) / (1.0 - x))
        else:
            fisher[i] = 0.0
        
        if i > 0:
            fisher_signal[i] = fisher[i - 1]
        else:
            fisher_signal[i] = fisher[i]
    
    return fisher, fisher_signal

def calculate_hma(close, period=21):
    """Hull Moving Average - for HTF trend"""
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

def calculate_atr(high, low, close, period=14):
    """Average True Range - for stoploss"""
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
    
    # Calculate and align 1d HMA for trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, period=21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    kama = calculate_kama(close, period=10, fast_period=2, slow_period=30)
    fisher, fisher_signal = calculate_fisher_transform(high, low, period=9)
    atr = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.28  # Discrete position size (between 0.25-0.30 proven range)
    
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
        if np.isnan(hma_1d_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama[i]) or np.isnan(fisher[i]) or np.isnan(fisher_signal[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF TREND BIAS (1d HMA) ===
        hma_1d_bull = close[i] > hma_1d_aligned[i]
        hma_1d_bear = close[i] < hma_1d_aligned[i]
        
        # === KAMA TREND FILTER ===
        kama_bull = close[i] > kama[i]
        kama_bear = close[i] < kama[i]
        
        # === FISHER REVERSAL SIGNAL (LOOSE thresholds for trade generation) ===
        # Long: Fisher crosses ABOVE -1.5 (oversold reversal)
        fisher_long = (fisher_signal[i] < -1.5) and (fisher[i] >= -1.5)
        # Short: Fisher crosses BELOW +1.5 (overbought reversal)
        fisher_short = (fisher_signal[i] > 1.5) and (fisher[i] <= 1.5)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        
        # Long entry: Fisher reversal + KAMA bull + 1d HMA bull
        if fisher_long and kama_bull and hma_1d_bull:
            desired_signal = SIZE
        
        # Short entry: Fisher reversal + KAMA bear + 1d HMA bear
        elif fisher_short and kama_bear and hma_1d_bear:
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