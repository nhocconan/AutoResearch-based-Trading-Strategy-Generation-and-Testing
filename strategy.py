#!/usr/bin/env python3
"""
Experiment #029: 12h Donchian Breakout + 1d SMA50 Trend + ATR Stop

HYPOTHESIS: Price channel breakouts work in both bull and bear markets.
By using ONLY 2 core conditions (Donchian breakout + SMA50 trend), this strategy
generates clean signals with minimal fee drag. ATR trailing stop manages risk.

WHY 12h: Slower than 6h/4h = fewer but higher-quality signals.
20-period Donchian on 12h = 10-day channel. Breakouts every 2-3 weeks per direction.

WHY IT WORKS IN BULL AND BEAR: Uses symmetrical channels so shorting breakdowns
works in bear. Long breakouts work in bull. No directional bias.

TARGET: 50-150 total over 4 years = 12-37/year. HARD MAX: 200.
Signal size: 0.25.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_sma50_atr_v1"
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

def generate_signals(prices):
    close = prices["close"].values
    high = prices["high"].values
    low = prices["low"].values
    n = len(close)
    
    # === Load HTF data ONCE ===
    df_1d = get_htf_data(prices, '1d')
    
    # 1d SMA50 for trend direction (pre-aligned to avoid look-ahead)
    sma_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_1d)
    
    # Local 12h ATR
    atr_14 = calculate_atr(high, low, close, period=14)
    
    # Donchian channels (20 periods = 10 days on 12h)
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_atr = 0.0
    stop_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Need Donchian(20) + SMA50 buffer
    
    for i in range(warmup, n):
        # Skip if indicators not ready
        if np.isnan(atr_14[i]) or atr_14[i] <= 0:
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        if np.isnan(sma_1d_aligned[i]):
            signals[i] = 0.0
            in_position = False
            position_side = 0
            continue
        
        # === TREND DIRECTION (1d SMA50) ===
        trend_up = close[i] > sma_1d_aligned[i]
        trend_down = close[i] < sma_1d_aligned[i]
        
        # Previous bar's Donchian values (use shift to avoid look-ahead)
        prev_donchian_high = donchian_high[i - 1] if i > 0 else 0
        prev_donchian_low = donchian_low[i - 1] if i > 0 else 0
        
        desired_signal = 0.0
        
        # === ENTRY LOGIC (2 conditions only) ===
        if not in_position:
            # LONG: Price breaks above previous high + trend is up
            if high[i] > prev_donchian_high and trend_up:
                desired_signal = SIZE
            
            # SHORT: Price breaks below previous low + trend is down
            elif low[i] < prev_donchian_low and trend_down:
                desired_signal = -SIZE
        
        # === STOPLOSS (2.5 ATR trailing) ===
        stoploss_triggered = False
        
        if in_position and position_side > 0:
            # Update highest and trailing stop
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stoploss_triggered = True
        
        if in_position and position_side < 0:
            # Update lowest and trailing stop
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stoploss_triggered = True
        
        if stoploss_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_atr = atr_14[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = close[i] - 2.5 * entry_atr
                else:
                    stop_price = close[i] + 2.5 * entry_atr
        else:
            if in_position:
                in_position = False
                position_side = 0
                entry_atr = 0.0
                stop_price = 0.0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
        
        signals[i] = desired_signal
    
    return signals