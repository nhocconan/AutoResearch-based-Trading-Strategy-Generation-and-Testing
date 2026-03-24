#!/usr/bin/env python3
"""
Experiment #231: 6h Primary + 1d/1w HTF — Fisher Transform Reversal Strategy

Hypothesis: Ehlers Fisher Transform excels at catching reversals in bear/range markets
(2025+ test period is bearish). Unlike RSI which lags, Fisher Transform normalizes
prices to Gaussian distribution, making extreme values clearer for mean reversion.

Strategy Logic:
- Fisher Transform (period=14): Enter long when Fisher crosses above -1.5 from below
  (oversold reversal). Enter short when Fisher crosses below +1.5 from above.
- 1d HMA(50): Major trend bias - prefer longs when price > 1d HMA, shorts when <
- 1w HMA(21): Secular trend filter - adds conviction when aligned with 1d
- ATR(14) trailing stop: 2.5x ATR from entry extreme

Why 6h: Middle ground between 4h (too many trades) and 12h (too few). Target 30-60
trades/year. Fisher Transform should generate more signals than RSI extremes.

Position sizing: 0.30 (30% of capital) - conservative given 2022 crash risk.
Stoploss: 2.5x ATR trailing from highest/lowest since entry.

Key difference from failed #223: Simpler entry logic (Fisher cross only + 1d bias),
no Keltner/KAMA complexity that caused whipsaws.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_fisher_reversal_1d1w_v1"
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

def calculate_fisher_transform(high, low, period=14):
    """
    Ehlers Fisher Transform - converts prices to Gaussian-like distribution
    Makes extreme values more identifiable for reversal trading.
    
    Formula:
    1. Normalize price: (2 * (close - min_low) / (max_high - min_low) - 1)
    2. Apply smoothing: 0.66 * prev_value + 0.33 * normalized
    3. Fisher: 0.5 * ln((1 + value) / (1 - value))
    
    Long signal: Fisher crosses above -1.5 from below (oversold reversal)
    Short signal: Fisher crosses below +1.5 from above (overbought reversal)
    """
    n = len(high)
    if n < period:
        return np.full(n, np.nan)
    
    # Use typical price for Fisher calculation
    typical = (high + low) / 2.0
    
    fisher = np.zeros(n)
    fisher[:] = np.nan
    
    # Track previous values for smoothing
    prev_value = 0.0
    
    for i in range(period, n):
        # Find highest high and lowest low over lookback period
        highest_high = np.max(high[i-period+1:i+1])
        lowest_low = np.min(low[i-period+1:i+1])
        
        price_range = highest_high - lowest_low
        if price_range < 1e-10:
            fisher[i] = 0.0
            continue
        
        # Normalize price to -1 to +1 range
        normalized = 2.0 * (typical[i] - lowest_low) / price_range - 1.0
        
        # Clamp to avoid division by zero in Fisher formula
        normalized = max(-0.999, min(0.999, normalized))
        
        # Apply exponential smoothing (Ehlers recommendation)
        smoothed_value = 0.66 * prev_value + 0.33 * normalized
        smoothed_value = max(-0.999, min(0.999, smoothed_value))
        prev_value = smoothed_value
        
        # Fisher Transform formula
        fisher[i] = 0.5 * np.log((1.0 + smoothed_value) / (1.0 - smoothed_value))
    
    return fisher

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load HTF data ONCE before loop (Rule 1 - CRITICAL)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate and align HTF HMAs
    hma_1d_raw = calculate_hma(df_1d['close'].values, 50)
    hma_1d = align_htf_to_ltf(prices, df_1d, hma_1d_raw)
    
    hma_1w_raw = calculate_hma(df_1w['close'].values, 21)
    hma_1w = align_htf_to_ltf(prices, df_1w, hma_1w_raw)
    
    # Calculate primary (6h) indicators
    fisher = calculate_fisher_transform(high, low, period=14)
    atr = calculate_atr(high, low, close, 14)
    hma_6h = calculate_hma(close, 21)
    
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size
    
    # Position tracking for stoploss
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    for i in range(60, n):  # Start after indicators are ready
        # Skip if indicators not ready
        if np.isnan(fisher[i]) or np.isnan(atr[i]) or atr[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(hma_1d[i]) or np.isnan(hma_1w[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # === HTF BIAS ===
        htf_1d_bull = close[i] > hma_1d[i]
        htf_1d_bear = close[i] < hma_1d[i]
        htf_1w_bull = close[i] > hma_1w[i]
        htf_1w_bear = close[i] < hma_1w[i]
        
        # === FISHER TRANSFORM SIGNALS ===
        # Long: Fisher crosses above -1.5 from below (oversold reversal)
        # Short: Fisher crosses below +1.5 from above (overbought reversal)
        fisher_cross_long = fisher[i] > -1.5 and fisher[i-1] <= -1.5
        fisher_cross_short = fisher[i] < 1.5 and fisher[i-1] >= 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # Long entry: Fisher cross + 1d trend alignment (1w optional boost)
        if fisher_cross_long:
            if htf_1d_bull:
                # 1d aligned - full size
                desired_signal = SIZE
            elif htf_1w_bull:
                # 1d neutral but 1w bull - reduced size
                desired_signal = SIZE * 0.7
        
        # Short entry: Fisher cross + 1d trend alignment
        if fisher_cross_short:
            if htf_1d_bear:
                # 1d aligned - full size
                desired_signal = -SIZE
            elif htf_1w_bear:
                # 1d neutral but 1w bear - reduced size
                desired_signal = -SIZE * 0.7
        
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
        if desired_signal >= SIZE * 0.9:
            final_signal = SIZE
        elif desired_signal <= -SIZE * 0.9:
            final_signal = -SIZE
        elif desired_signal >= SIZE * 0.6:
            final_signal = SIZE * 0.7
        elif desired_signal <= -SIZE * 0.6:
            final_signal = -SIZE * 0.7
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