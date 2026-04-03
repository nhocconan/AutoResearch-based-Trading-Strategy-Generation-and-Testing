#!/usr/bin/env python3
"""
Experiment #171: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot direction (from 1d timeframe) 
capture institutional swing trades with low whipsaw. Weekly pivot levels (calculated from 
prior week's OHLC) provide structural support/resistance that works in both bull/bear markets. 
Volume confirmation (1.5x average) ensures participation. Discrete sizing (0.25) and ATR 
trailing stop (2.5x) manage risk. Targets 15-25 trades/year on 6h timeframe to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def calculate_wma(data, window):
    """Weighted Moving Average"""
    if len(data) < window:
        return np.full(len(data), np.nan)
    weights = np.arange(1, window + 1, dtype=np.float64)
    return np.convolve(data, weights[::-1], mode='valid') / weights.sum()

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
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    df_1d_index = df_1d.index  # DatetimeIndex
    
    # Calculate weekly pivot points from prior week's OHLC
    # We need to group by week and calculate: PP = (H+L+C)/3, R1 = 2*PP - L, S1 = 2*PP - H
    # Then align to 6h timeframe
    
    # Resample 1d data to weekly OHLC (using actual week boundaries)
    weekly_ohlc = df_1d.resample('W-WED', label='left', closed='left').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last'
    }).shift(1)  # Use prior week's data to avoid look-ahead
    
    # Calculate pivot points for each week
    weekly_ohlc['PP'] = (weekly_ohlc['high'] + weekly_ohlc['low'] + weekly_ohlc['close']) / 3.0
    weekly_ohlc['R1'] = 2 * weekly_ohlc['PP'] - weekly_ohlc['low']
    weekly_ohlc['S1'] = 2 * weekly_ohlc['PP'] - weekly_ohlc['high']
    weekly_ohlc['R2'] = weekly_ohlc['PP'] + (weekly_ohlc['high'] - weekly_ohlc['low'])
    weekly_ohlc['S2'] = weekly_ohlc['PP'] - (weekly_ohlc['high'] - weekly_ohlc['low'])
    weekly_ohlc['R3'] = weekly_ohlc['high'] + 2 * (weekly_ohlc['PP'] - weekly_ohlc['low'])
    weekly_ohlc['S3'] = weekly_ohlc['low'] - 2 * (weekly_ohlc['high'] - weekly_ohlc['PP'])
    
    # Forward fill weekly values to daily index (each day gets prior week's pivot)
    weekly_pivots = weekly_ohlc[['PP', 'R1', 'S1', 'R2', 'S2', 'R3', 'S3']].ffill()
    
    # Align weekly pivots to 1d timeframe (each day gets the weekly pivot from prior week)
    df_1d = df_1d.join(weekly_pivots, how='left')
    
    # Extract pivot values as arrays
    pp_1d = df_1d['PP'].values
    r1_1d = df_1d['R1'].values
    s1_1d = df_1d['S1'].values
    r2_1d = df_1d['R2'].values
    s2_1d = df_1d['S2'].values
    r3_1d = df_1d['R3'].values
    s3_1d = df_1d['S3'].values
    
    # Align 1d pivot values to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # === 6h Indicators ===
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
            np.isnan(vol_ma_20[i]) or np.isnan(pp_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Weekly Pivot Direction Filter ---
        # Price above weekly pivot = bullish bias, below = bearish bias
        price_vs_pivot = close[i] - pp_aligned[i]
        pivot_bullish = price_vs_pivot > 0
        pivot_bearish = price_vs_pivot < 0
        
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
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR breaks below weekly pivot
                    if close[i] <= dc_lower_20[i] or close[i] < pp_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR breaks above weekly pivot
                    if close[i] >= dc_upper_20[i] or close[i] > pp_aligned[i]:
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
        # Breakout above upper Donchian with bullish weekly pivot bias and volume confirmation
        if bullish_breakout and pivot_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with bearish weekly pivot bias and volume confirmation
        elif bearish_breakout and pivot_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals