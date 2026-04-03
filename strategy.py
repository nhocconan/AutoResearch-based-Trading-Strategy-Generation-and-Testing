#!/usr/bin/env python3
"""
Experiment #070: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Spike

HYPOTHESIS: Daily Donchian breakouts aligned with weekly HMA trend capture institutional order flow with minimal fee drag.
Weekly HMA provides smooth trend filter to avoid whipsaws. Volume confirmation (2.0x average) ensures follow-through.
Designed for 15-25 trades/year to minimize fee drift while maintaining statistical significance.
Uses discrete position sizing (0.25) to reduce churn. Works in both bull/bear markets by trading breakouts
in direction of weekly HMA slope.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_1w_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def calculate_hma(close, period):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    half = period // 2
    sqrt_n = int(np.sqrt(period))
    
    def wma(arr, window):
        if len(arr) < window:
            return np.full(len(arr), np.nan)
        weights = np.arange(1, window + 1, dtype=np.float64)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    
    close_series = pd.Series(close)
    wma_half = wma(close_series.values, half)
    wma_full = wma(close_series.values, period)
    
    # Align arrays: wma_half starts at index (period - half), wma_full at (period - 1)
    raw_2wma = np.full(n, np.nan)
    raw_2wma[half-1:half-1+len(wma_half)*2] = wma_half * 2
    raw_2wma[period-1:period-1+len(wma_full)] -= wma_full
    
    wma_2wma = wma(raw_2wma[~np.isnan(raw_2wma)], sqrt_n)
    
    hma = np.full(n, np.nan)
    start_idx = period - 1 + (sqrt_n - 1)//2
    end_idx = start_idx + len(wma_2wma)
    hma[start_idx:end_idx] = wma_2wma
    
    return hma

def calculate_atr(high, low, close, period=14):
    """Average True Range for stoploss calculation."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    
    tr = np.zeros(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).ewm(span=period, min_periods=period, adjust=False).mean().values
    return atr

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    hma_21 = calculate_hma(df_1w['close'].values, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # === 1d Indicators ===
    atr_14 = calculate_atr(high, low, close, period=14)
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
            np.isnan(vol_ma_20[i]) or np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly HMA Trend ---
        # Calculate slope of HMA over 3 periods
        if i >= 3:
            hma_slope = hma_21_aligned[i] - hma_21_aligned[i-3]
            hma_bullish = hma_slope > 0
            hma_bearish = hma_slope < 0
        else:
            hma_bullish = False
            hma_bearish = False
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2.0x volume spike
        
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
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~3d)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR HMA turns bearish
                    if close[i] <= dc_lower_20[i] or not hma_bullish:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR HMA turns bullish
                    if close[i] >= dc_upper_20[i] or not hma_bearish:
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
        # Breakout above upper Donchian with bullish weekly HMA and volume confirmation
        if bullish_breakout and hma_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with bearish weekly HMA and volume confirmation
        elif bearish_breakout and hma_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals