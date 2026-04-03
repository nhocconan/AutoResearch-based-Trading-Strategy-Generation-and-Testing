#!/usr/bin/env python3
"""
Experiment #136: 12h Donchian(20) Breakout + 1d HMA Trend + Volume Confirmation

HYPOTHESIS: 12h Donchian breakouts aligned with 1d HMA trend capture swing momentum
with lower whipsaw than using 1h/4h HTF. 1d HMA (21) filters for primary trend direction
while being responsive to regime changes. Volume confirmation (1.5x average) ensures
institutional participation. ATR trailing stop (2.5x) manages risk. Targets 12-37 trades/year
on 12h timeframe to minimize fee drag. Works in bull/bear markets by trading breakouts
in direction of 1d HMA trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    def wma(data, window):
        if len(data) < window:
            return np.full(len(data), np.nan)
        weights = np.arange(1, window + 1, dtype=np.float64)
        return np.convolve(data, weights[::-1], mode='valid') / weights.sum()
    
    half = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close, half)
    wma_full = wma(close, period)
    
    # 2*WMA(half) - WMA(full)
    diff = 2 * np.concatenate([np.full(half - 1, np.nan), wma_half]) - np.concatenate([np.full(period - 1, np.nan), wma_full])
    
    # WMA of diff with sqrt_period
    hma = wma(diff, sqrt_period)
    # Adjust for padding
    hma = np.concatenate([np.full(sqrt_period - 1, np.nan), hma])
    
    return hma

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === 12h Indicators ===
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(hma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- 1d HMA Trend ---
        hma_bullish = close[i] > hma_1d_aligned[i]
        hma_bearish = close[i] < hma_1d_aligned[i]
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~1day)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR breaks below HMA
                    if close[i] <= dc_lower_20[i] or close[i] < hma_1d_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR breaks above HMA
                    if close[i] >= dc_upper_20[i] or close[i] > hma_1d_aligned[i]:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: 
        # Breakout above upper Donchian with bullish 1d HMA trend and volume confirmation
        if bullish_breakout and hma_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with bearish 1d HMA trend and volume confirmation
        elif bearish_breakout and hma_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals