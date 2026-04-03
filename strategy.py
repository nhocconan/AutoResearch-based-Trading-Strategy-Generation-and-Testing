#!/usr/bin/env python3
"""
Experiment #213: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Confirmation

HYPOTHESIS: 4h Donchian breakouts aligned with 12h HMA trend capture medium-term momentum with reduced whipsaw. Volume confirmation ensures breakouts have institutional participation. The 12h HMA acts as a trend filter to avoid counter-trend entries. Targets 19-50 trades/year on 4h timeframe to balance opportunity with fee drag. Works in both bull and bear markets by only taking breakouts in the direction of the 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_volume_12h_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) >= 21:
        # Calculate HMA(21) on 12h close
        hma_12h = calculate_hma(df_12h['close'].values, 21)
        # Align to 4h timeframe
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
        # Trend: 1 if close > HMA, -1 if close < HMA
        hma_trend = np.where(close[:len(hma_12h_aligned)] > hma_12h_aligned, 1, -1)
    else:
        hma_trend = np.full(n, 0)  # Neutral if insufficient data
    
    # === 4h Indicators ===
    # ATR(14) for stoploss and volatility filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian Channel(20) - shift(1) to avoid look-ahead
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume MA(20) for confirmation
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
            np.isnan(vol_ma_20[i]) or i >= len(hma_trend)):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Trend Filter from 12h HMA ---
        trend_ok_long = hma_trend[i] > 0   # 12h trend bullish
        trend_ok_short = hma_trend[i] < 0  # 12h trend bearish
        
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
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR 12h trend turns bearish
                    if close[i] <= dc_lower_20[i] or hma_trend[i] < 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR 12h trend turns bullish
                    if close[i] >= dc_upper_20[i] or hma_trend[i] > 0:
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
        # Breakout above upper Donchian with volume confirmation and bullish 12h trend
        if bullish_breakout and vol_ok and trend_ok_long:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation and bearish 12h trend
        elif bearish_breakout and vol_ok and trend_ok_short:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

def calculate_hma(arr, period):
    """Calculate Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.zeros_like(arr)
    for i in range(half_period, len(arr)):
        weights = np.arange(1, half_period + 1)
        wma_half[i] = np.dot(arr[i - half_period + 1:i + 1], weights) / weights.sum()
    
    # WMA of full period
    wma_full = np.zeros_like(arr)
    for i in range(period, len(arr)):
        weights = np.arange(1, period + 1)
        wma_full[i] = np.dot(arr[i - period + 1:i + 1], weights) / weights.sum()
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of raw_hma with sqrt_period
    hma = np.zeros_like(arr)
    for i in range(sqrt_period, len(arr)):
        weights = np.arange(1, sqrt_period + 1)
        hma[i] = np.dot(raw_hma[i - sqrt_period + 1:i + 1], weights) / weights.sum()
    
    return hma