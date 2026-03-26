#!/usr/bin/env python3
"""
Experiment #008: 12h Weekly Donchian Breakout

HYPOTHESIS: 12h timeframe with simple dual-confirmation:
- 1w Donchian for trend direction (bull if above 20w high, bear if below)
- 12h Donchian(20) for entry timing
- ATR for volatility-adaptive position management

WHY THIS SHOULD WORK:
- Donchian breakout is one of the few patterns with POSITIVE edge historically
- Weekly filter removes counter-trend trades (major failure mode in bear 2022)
- 12h allows institutional participation while staying liquid
- Simple = fewer conflicting signals = consistent trade count

TARGET: 75-150 total trades over 4 years (19-37/year).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_weekly_donchian_simple_v1"
timeframe = "12h"
leverage = 1.0

def calculate_atr(high, low, close, period=14):
    """Average True Range"""
    n = len(close)
    if n < period + 1:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def calculate_donchian(high, low, period=20):
    """Donchian Channel - upper = highest high, lower = lowest low"""
    n = len(high)
    upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
    return upper, lower

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly Donchian(20) for trend direction
    weekly_upper, weekly_lower = calculate_donchian(
        df_1w['high'].values, df_1w['low'].values, period=20
    )
    
    # Align weekly to 12h (shifted by 1 to avoid look-ahead)
    weekly_upper_aligned = align_htf_to_ltf(prices, df_1w, weekly_upper)
    weekly_lower_aligned = align_htf_to_ltf(prices, df_1w, weekly_lower)
    
    # 12h indicators
    upper_12h, lower_12h = calculate_donchian(high, low, period=20)
    atr_14 = calculate_atr(high, low, close, period=14)
    
    signals = np.zeros(n)
    SIZE = 0.30
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: need 20 bars for Donchian + ATR
    warmup = 50
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(upper_12h[i]) or np.isnan(lower_12h[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(atr_14[i]) or atr_14[i] <= 1e-10:
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        if np.isnan(weekly_upper_aligned[i]) or np.isnan(weekly_lower_aligned[i]):
            signals[i] = 0.0
            if in_position:
                in_position = False
                position_side = 0
            continue
        
        # Weekly trend direction
        weekly_bullish = close[i] > weekly_upper_aligned[i]
        weekly_bearish = close[i] < weekly_lower_aligned[i]
        
        # 12h breakout signals
        bull_breakout = close[i] > upper_12h[i]  # Price breaks above 12h high
        bear_breakout = close[i] < lower_12h[i]  # Price breaks below 12h low
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: 12h breakout + weekly confirms bullish
        if bull_breakout and weekly_bullish:
            desired_signal = SIZE
        
        # SHORT: 12h breakdown + weekly confirms bearish
        if bear_breakout and weekly_bearish:
            desired_signal = -SIZE
        
        # === STOPLOSS CHECK (2x ATR) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === OPPOSITE SIGNAL EXIT ===
        # If we get a bearish breakout while long, exit
        if in_position and position_side > 0 and bear_breakout:
            desired_signal = 0.0
        
        # If we get a bullish breakout while short, exit
        if in_position and position_side < 0 and bull_breakout:
            desired_signal = 0.0
        
        # === UPDATE POSITION TRACKING ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
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
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals