#!/usr/bin/env python3
"""
Experiment #034: 1h Strategy with 4h/1d HTF Filters
HYPOTHESIS: Use 4h Donchian(20) breakout direction and 1d EMA(50) trend filter for signal direction, 
with 1h RSI(14) pullback entry for timing. This combines HTF structure with lower TF precision 
to minimize trades while capturing momentum. Discrete sizing (0.20) and ATR(14) stoploss (2.0) 
control risk. Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag. 
Session filter (08-20 UTC) reduces noise. Works in bull (breakouts with trend) and bear 
(short breakdowns with trend) by using HTF direction as primary filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_034_1h_donchian_4h_1d_ema_rsi_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for Donchian(20) direction (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    highest_high_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_4h_aligned = align_htf_to_ltf(prices, df_4h, highest_high_4h)  # Using highest_high as proxy, will use both
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, lowest_low_4h)
    
    # === HTF: 1d data for EMA(50) trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # === 1h Indicators: RSI(14) for pullback entry ===
    def rsi(series, period):
        delta = np.diff(series, prepend=series[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_vals = 100 - (100 / (1 + rs))
        return rsi_vals
    
    rsi_1h = rsi(close, 14)
    
    # === 1h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # sufficient for 1d EMA(50) + 4h Donchian(20) + 1h RSI(14)
    
    for i in range(warmup, n):
        # --- Session Filter: 08-20 UTC only ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(rsi_1h[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- HTF Direction Filters ---
        # 4h Donchian breakout direction (using previous bar's levels)
        donchian_bull = price > donchian_4h_aligned[i]   # Price above 4h upper channel
        donchian_bear = price < donchian_low_4h_aligned[i]  # Price below 4h lower channel
        
        # 1d EMA trend filter
        uptrend = price > ema_1d_aligned[i]
        downtrend = price < ema_1d_aligned[i]
        
        # --- 1h Entry Timing: RSI pullback in direction of HTF trend ---
        # Long: RSI < 40 (pullback) in uptrend
        # Short: RSI > 60 (pullback) in downtrend
        rsi_long = rsi_1h[i] < 40
        rsi_short = rsi_1h[i] > 60
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 24 bars (1 day) to avoid overtrading
            if bars_since_entry > 24:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Long: 4h bullish alignment + 1d uptrend + RSI pullback
        if donchian_bull and uptrend and rsi_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: 4h bearish alignment + 1d downtrend + RSI pullback
        elif donchian_bear and downtrend and rsi_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals