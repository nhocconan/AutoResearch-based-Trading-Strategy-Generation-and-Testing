#!/usr/bin/env python3
"""
Experiment #1029: 4h Primary + 1d HTF — Simplified Fisher + Choppiness Regime

Hypothesis: After analyzing 746+ failed strategies, the #1 failure mode is TOO MANY
filters causing 0 trades (Sharpe=0.000). This strategy SIMPLIFIES the entry logic
while keeping the core edge from Fisher Transform + Choppiness regime detection.

Key changes from failed #1014:
1. SINGLE HTF filter (1d HMA only, not 12h+1d) - reduces filter conflicts
2. RELAXED Fisher thresholds (-1.0/+1.0 not -1.5/+1.5) - more trade signals
3. SIMPLIFIED Choppiness (single threshold at 50, not 38.2/61.8) - clearer regime
4. REMOVED vol crush exit logic - was killing winning trades prematurely
5. SIMPLIFIED hold logic - maintain position unless clear exit signal

Core Edge:
- Fisher Transform catches reversals in bear/range markets (2025 test period)
- Choppiness Index switches between mean-revert (chop>50) and trend-follow (chop<50)
- 1d HMA21 provides long-term trend bias (long above, short below)
- ATR 2.5x trailing stop for risk management

Target: 30-50 trades/year on 4h, Sharpe > 0.612, ALL symbols positive
Timeframe: 4h
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_fisher_chop_simple_1d_hma_atr_v1"
timeframe = "4h"
leverage = 1.0

def calculate_fisher_transform(high, low, close, period=9):
    """
    Ehlers Fisher Transform - converts price to Gaussian distribution.
    Entry: Fisher crosses above -1.0 (oversold reversal)
    Exit: Fisher crosses below +1.0 (overbought reversal)
    """
    n = len(close)
    fisher = np.full(n, np.nan)
    trigger = np.full(n, np.nan)
    
    if n < period + 5:
        return fisher, trigger
    
    for i in range(period, n):
        highest = np.max(high[i-period+1:i+1])
        lowest = np.min(low[i-period+1:i+1])
        
        if highest == lowest:
            fisher[i] = 0.0
            trigger[i] = fisher[i]
            continue
        
        value = (2.0 * close[i] - highest - lowest) / (highest - lowest + 1e-10)
        value = np.clip(value * 0.999, -0.999, 0.999)
        
        fisher[i] = 0.5 * np.log((1.0 + value) / (1.0 - value + 1e-10))
        trigger[i] = fisher[i-1] if i > period else fisher[i]
    
    return fisher, trigger

def calculate_choppiness(high, low, close, period=14):
    """
    Choppiness Index - measures trending vs ranging.
    CHOP > 50 = ranging (mean reversion)
    CHOP < 50 = trending (trend follow)
    """
    n = len(close)
    chop = np.full(n, np.nan)
    
    if n < period + 1:
        return chop
    
    for i in range(period, n):
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        atr_sum = 0.0
        for j in range(i-period+1, i+1):
            if j == 0:
                tr = high[j] - low[j]
            else:
                tr = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_sum += tr
        
        if atr_sum > 0 and (highest_high - lowest_low) > 0:
            chop[i] = 100 * np.log10(atr_sum / (highest_high - lowest_low)) / np.log10(period)
        else:
            chop[i] = 50
    
    return np.clip(chop, 0, 100)

def calculate_atr(high, low, close, period=14):
    """Average True Range with proper min_periods."""
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_hma(series, period):
    """Hull Moving Average."""
    series = pd.Series(series)
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = series.rolling(window=half, min_periods=half).mean() * 2
    wma_full = series.rolling(window=period, min_periods=period).mean()
    
    wma_diff = wma_half - wma_full
    hma = wma_diff.rolling(window=sqrt_period, min_periods=sqrt_period).mean()
    
    return hma.values

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate and align 1d HMA21 for long-term trend bias
    hma_1d_raw = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    # Calculate primary (4h) indicators
    fisher_4h, trigger_4h = calculate_fisher_transform(high, low, close, period=9)
    atr_4h = calculate_atr(high, low, close, period=14)
    chop_4h = calculate_choppiness(high, low, close, period=14)
    
    signals = np.zeros(n)
    BASE_SIZE = 0.30
    REDUCED_SIZE = 0.20
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(100, n):
        # Skip if indicators not ready
        if np.isnan(fisher_4h[i]) or np.isnan(trigger_4h[i]):
            continue
        if np.isnan(atr_4h[i]) or atr_4h[i] <= 1e-10:
            continue
        if np.isnan(hma_1d_aligned[i]) or np.isnan(chop_4h[i]):
            continue
        
        # === TREND BIAS (1d HMA21) ===
        above_hma = close[i] > hma_1d_aligned[i]
        below_hma = close[i] < hma_1d_aligned[i]
        
        # === REGIME (Choppiness) ===
        is_choppy = chop_4h[i] > 50  # Mean reversion regime
        is_trending = chop_4h[i] <= 50  # Trend follow regime
        
        # === FISHER SIGNALS ===
        fisher_cross_long = fisher_4h[i] > -1.0 and trigger_4h[i] <= -1.0
        fisher_cross_short = fisher_4h[i] < 1.0 and trigger_4h[i] >= 1.0
        fisher_oversold = fisher_4h[i] < -1.0
        fisher_overbought = fisher_4h[i] > 1.0
        
        desired_signal = 0.0
        
        # === LONG ENTRIES ===
        if is_choppy:
            # Mean reversion: buy oversold in choppy market
            if fisher_cross_long and above_hma:
                desired_signal = BASE_SIZE
            elif fisher_oversold and above_hma:
                desired_signal = REDUCED_SIZE
        else:
            # Trend following: buy pullbacks in trending market
            if fisher_cross_long:
                desired_signal = BASE_SIZE
            elif fisher_4h[i] > -0.5 and fisher_4h[i-1] <= -0.5 and above_hma:
                desired_signal = REDUCED_SIZE
        
        # === SHORT ENTRIES ===
        if is_choppy:
            # Mean reversion: sell overbought in choppy market
            if fisher_cross_short and below_hma:
                desired_signal = -BASE_SIZE
            elif fisher_overbought and below_hma:
                desired_signal = -REDUCED_SIZE
        else:
            # Trend following: sell rallies in trending market
            if fisher_cross_short:
                desired_signal = -BASE_SIZE
            elif fisher_4h[i] < 0.5 and fisher_4h[i-1] >= 0.5 and below_hma:
                desired_signal = -REDUCED_SIZE
        
        # === STOPLOSS (Trailing ATR 2.5x) ===
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
        
        # === EXIT CONDITIONS ===
        # Exit long if trend reverses or Fisher extreme overbought
        if in_position and position_side > 0:
            if below_hma and fisher_4h[i] > 0.5:
                desired_signal = 0.0
            if fisher_4h[i] > 1.5:
                desired_signal = 0.0
        
        # Exit short if trend reverses or Fisher extreme oversold
        if in_position and position_side < 0:
            if above_hma and fisher_4h[i] < -0.5:
                desired_signal = 0.0
            if fisher_4h[i] < -1.5:
                desired_signal = 0.0
        
        # === HOLD LOGIC ===
        if in_position and desired_signal == 0.0 and not stoploss_triggered:
            if position_side > 0 and above_hma and fisher_4h[i] < 1.0:
                desired_signal = BASE_SIZE
            elif position_side < 0 and below_hma and fisher_4h[i] > -1.0:
                desired_signal = -BASE_SIZE
        
        # === DISCRETIZE SIGNAL ===
        if desired_signal > 0:
            desired_signal = BASE_SIZE if desired_signal >= 0.25 else REDUCED_SIZE
        elif desired_signal < 0:
            desired_signal = -BASE_SIZE if desired_signal <= -0.25 else -REDUCED_SIZE
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
                highest_since_entry = close[i] if position_side > 0 else 0.0
                lowest_since_entry = close[i] if position_side < 0 else float('inf')
            elif np.sign(desired_signal) != position_side:
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_4h[i]
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
        
        signals[i] = desired_signal
    
    return signals