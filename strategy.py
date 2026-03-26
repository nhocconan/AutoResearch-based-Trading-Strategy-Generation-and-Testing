#!/usr/bin/env python3
"""
Experiment #010: 1d Donchian Breakout + Vol Spike + 1w Trend Filter

HYPOTHESIS: Donchian breakout on 1d captures medium-term structural shifts.
Volume spike confirms institutional involvement. 1w EMA(21) filter ensures
we trade with the larger trend structure.

WHY THIS SHOULD WORK IN BOTH BULL AND BEAR:
- Breakout trading works in ALL market conditions
- Bull markets: breakouts above 1w EMA = high-probability longs
- Bear markets: breakouts below 1w EMA = high-probability shorts
- 1d timeframe = fewer trades = less fee drag than 4h/6h
- ATR-based stops adapt to volatility in any market regime

TARGET: 40-80 total trades over 4 years (10-20/year).
KEY: Simple 3-condition entry (breakout + trend + vol).
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_vol_1w_v1"
timeframe = "1d"
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
    volume = prices["volume"].values
    n = len(close)
    
    # Load 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    ema_1w_raw = pd.Series(df_1w['close'].values).ewm(span=21, min_periods=21, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_raw)
    
    # Daily indicators
    atr = calculate_atr(high, low, close, 14)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    
    warmup = 70
    
    for i in range(warmup, n):
        # Check indicators ready
        if np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(donch_high[i]) or np.isnan(ema_1w_aligned[i]):
            if in_position:
                stop_triggered = False
                if position_side > 0 and low[i] < stop_price:
                    stop_triggered = True
                elif position_side < 0 and high[i] > stop_price:
                    stop_triggered = True
                if stop_triggered:
                    in_position = False
                    position_side = 0
            signals[i] = 0.0
            continue
        
        # Volume ratio
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 1.0
        
        # === TREND FILTER (1w EMA) ===
        bullish_1w = close[i] > ema_1w_aligned[i]
        bearish_1w = close[i] < ema_1w_aligned[i]
        
        # Volume spike
        vol_spike = vol_ratio > 1.5
        
        # === ENTRY LOGIC ===
        desired_signal = 0.0
        
        # LONG: Upper Donchian breakout + bullish weekly + vol spike
        if close[i] > donch_high[i] and bullish_1w and vol_spike:
            desired_signal = SIZE
        
        # SHORT: Lower Donchian breakout + bearish weekly + vol spike
        if close[i] < donch_low[i] and bearish_1w and vol_spike:
            desired_signal = -SIZE
        
        # === STOPLOSS ===
        stop_triggered = False
        
        if in_position and position_side > 0:
            highest_since_entry = max(highest_since_entry, high[i])
            trailing_stop = highest_since_entry - 2.5 * entry_atr
            stop_price = max(stop_price, trailing_stop)
            if low[i] < stop_price:
                stop_triggered = True
        
        if in_position and position_side < 0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            trailing_stop = lowest_since_entry + 2.5 * entry_atr
            stop_price = min(stop_price, trailing_stop)
            if high[i] > stop_price:
                stop_triggered = True
        
        if stop_triggered:
            desired_signal = 0.0
        
        # === TAKE PROFIT (3:1 ratio) ===
        tp_triggered = False
        
        if in_position and position_side > 0:
            tp_price = entry_price + 3.0 * entry_atr
            if high[i] >= tp_price:
                tp_triggered = True
        
        if in_position and position_side < 0:
            tp_price = entry_price - 3.0 * entry_atr
            if low[i] <= tp_price:
                tp_triggered = True
        
        if tp_triggered:
            desired_signal = 0.0
        
        # === UPDATE POSITION ===
        if desired_signal != 0.0:
            if not in_position or np.sign(desired_signal) != position_side:
                in_position = True
                position_side = int(np.sign(desired_signal))
                entry_price = close[i]
                entry_atr = atr[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                if position_side > 0:
                    stop_price = entry_price - 2.5 * entry_atr
                else:
                    stop_price = entry_price + 2.5 * entry_atr
        elif in_position:
            if position_side > 0:
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = max(stop_price, highest_since_entry - 2.5 * entry_atr)
            else:
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = min(stop_price, lowest_since_entry + 2.5 * entry_atr)
        
        if desired_signal == 0.0 and in_position:
            in_position = False
            position_side = 0
        
        signals[i] = desired_signal
    
    return signals