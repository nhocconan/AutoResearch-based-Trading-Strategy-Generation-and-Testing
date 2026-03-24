#!/usr/bin/env python3
"""
Experiment #014: 4h Primary + 12h HTF — Fisher Transform + KAMA Trend + Volume

Hypothesis: Previous dual-regime strategies over-filtered entries. This uses:
1. Ehlers Fisher Transform (period=9) - superior reversal detection in bear/range markets
2. KAMA (Kaufman Adaptive MA) - adapts to volatility, less whipsaw than EMA/HMA
3. 12h KAMA for trend bias (not hard filter) - asymmetric sizing
4. Volume confirmation - only trade when volume > 1.5x 20-period average
5. ATR trailing stop at 2.5x for risk management

Why this should work:
- Fisher Transform catches reversals better than RSI in 2022 crash and 2025 bear
- KAMA reduces whipsaw in choppy conditions (BTC/ETH specialty)
- Volume filter ensures we only trade confirmed moves (reduces false signals)
- Looser Fisher thresholds (-1.8/+1.8 instead of -1.5/+1.5) for more trades
- 12h bias for sizing, not binary filter (ensures trade generation)

Entry Logic:
- Long: Fisher < -1.8 + crosses above -1.5 + price > KAMA(4h) + volume confirm
- Short: Fisher > +1.8 + crosses below +1.5 + price < KAMA(4h) + volume confirm
- Size: 0.30 with 12h trend, 0.20 against 12h trend
- Stoploss: 2.5x ATR trailing

Target: Sharpe > 0.4, trades > 30/symbol train, > 3/symbol test, DD > -40%
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_kama_volume_12h_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, period=9):
    """
    Ehlers Fisher Transform
    Converts price to Gaussian distribution for clearer reversal signals
    Long: Fisher crosses above -1.5 from below -1.8
    Short: Fisher crosses below +1.5 from above +1.8
    """
    n = len(high)
    if n < period * 2:
        return np.full(n, np.nan)
    
    fisher = np.full(n, np.nan)
    fisher_prev = np.full(n, np.nan)
    
    # Typical price
    typical = (high + low + high) / 3.0  # Use high twice for emphasis
    
    # Normalize price to 0-1 range
    for i in range(period, n):
        highest = np.max(high[i - period + 1:i + 1])
        lowest = np.min(low[i - period + 1:i + 1])
        
        if highest == lowest:
            continue
        
        # Normalize
        normalized = 2.0 * (typical[i] - lowest) / (highest - lowest) - 1.0
        
        # Clamp to avoid division issues
        normalized = np.clip(normalized, -0.99, 0.99)
        
        # Fisher calculation
        fisher_val = 0.5 * np.log((1.0 + normalized) / (1.0 - normalized))
        
        # Smooth with previous value
        if i > period and not np.isnan(fisher[i-1]):
            fisher[i] = 0.6 * fisher_val + 0.4 * fisher[i-1]
            fisher_prev[i] = fisher[i-1]
        else:
            fisher[i] = fisher_val
            fisher_prev[i] = fisher_val
    
    return fisher, fisher_prev

def calculate_kama(close, period=10, fast_period=2, slow_period=30):
    """
    Kaufman Adaptive Moving Average (KAMA)
    Adapts smoothing based on market efficiency/volatility
    Less whipsaw than EMA in choppy markets
    """
    n = len(close)
    if n < period + slow_period:
        return np.full(n, np.nan)
    
    kama = np.full(n, np.nan)
    
    # Calculate Efficiency Ratio (ER)
    er = np.zeros(n)
    for i in range(period, n):
        signal = abs(close[i] - close[i - period])
        noise = np.sum(np.abs(np.diff(close[i - period:i + 1])))
        if noise > 1e-10:
            er[i] = signal / noise
        else:
            er[i] = 1.0
    
    # Calculate smoothing constant
    fast_sc = 2.0 / (fast_period + 1.0)
    slow_sc = 2.0 / (slow_period + 1.0)
    
    # Initialize KAMA
    kama[period] = close[period]
    
    for i in range(period + 1, n):
        sc = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])
    
    return kama

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

def calculate_volume_sma(volume, period=20):
    """Volume Simple Moving Average"""
    n = len(volume)
    if n < period:
        return np.full(n, np.nan)
    
    vol_sma = pd.Series(volume).rolling(window=period, min_periods=period).mean().values
    return vol_sma

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    volume = prices["volume"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate and align 12h KAMA for trend bias
    kama_12h_raw = calculate_kama(df_12h['close'].values, period=10)
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h_raw)
    
    # Calculate primary (4h) indicators
    fisher, fisher_prev = calculate_fisher_transform(high, low, period=9)
    kama_4h = calculate_kama(close, period=10)
    atr = calculate_atr(high, low, close, period=14)
    vol_sma = calculate_volume_sma(volume, period=20)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
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
        if np.isnan(kama_12h_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(fisher[i]) or np.isnan(fisher_prev[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        if np.isnan(kama_4h[i]) or np.isnan(vol_sma[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === VOLUME CONFIRMATION ===
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        # === HTF TREND BIAS (12h KAMA) ===
        hma_12h_bull = close[i] > kama_12h_aligned[i]
        hma_12h_bear = close[i] < kama_12h_aligned[i]
        
        # === PRIMARY TREND (4h KAMA) ===
        kama_4h_bull = close[i] > kama_4h[i]
        kama_4h_bear = close[i] < kama_4h[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher was < -1.8, now crosses above -1.5
        fisher_long = (fisher_prev[i] < -1.8) and (fisher[i] > -1.5)
        
        # Short: Fisher was > +1.8, now crosses below +1.5
        fisher_short = (fisher_prev[i] > 1.8) and (fisher[i] < 1.5)
        
        # === DESIRED SIGNAL ===
        desired_signal = 0.0
        signal_strength = 0.0
        
        # Long entry
        if fisher_long and kama_4h_bull and volume_confirm:
            if hma_12h_bull:
                signal_strength = BASE_SIZE
            else:
                signal_strength = REDUCED_SIZE
            desired_signal = signal_strength
        
        # Short entry
        elif fisher_short and kama_4h_bear and volume_confirm:
            if hma_12h_bear:
                signal_strength = BASE_SIZE
            else:
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
        
        if desired_signal >= BASE_SIZE * 0.85:
            final_signal = BASE_SIZE
        elif desired_signal <= -BASE_SIZE * 0.85:
            final_signal = -BASE_SIZE
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