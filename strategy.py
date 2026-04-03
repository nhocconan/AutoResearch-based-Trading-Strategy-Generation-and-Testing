#!/usr/bin/env python3
"""
Experiment #005: 12h Donchian(20) Breakout + 1d Camarilla Pivot + Volume Spike

HYPOTHESIS: Combining 12h Donchian breakouts with 1d Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) 
and volume confirmation creates a high-probability entry signal. The strategy trades breakouts above R4 or below S4 
with 1d trend alignment, and fades at R3/S3 with counter-trend entries when 1d trend is weak. Designed to capture 
both trending and mean-reverting moves in BTC/ETH/SOL with controlled trade frequency (~12-37/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_camarilla_volume_v1"
timeframe = "12h"
leverage = 1.0

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
    
    # === HTF: 1d OHLC for Camarilla pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    # Previous day's OHLC for today's Camarilla levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla pivot levels (based on previous day)
    camarilla_r4 = prev_close + 1.5 * prev_range
    camarilla_r3 = prev_close + 1.125 * prev_range
    camarilla_s3 = prev_close - 1.125 * prev_range
    camarilla_s4 = prev_close - 1.5 * prev_range
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Align HTF arrays to LTF (12h) with shift(1) for completed bars only
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # === 12h Indicators ===
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
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or np.isnan(camarilla_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Levels ---
        camarilla_r4 = camarilla_r4_aligned[i]
        camarilla_r3 = camarilla_r3_aligned[i]
        camarilla_s3 = camarilla_s3_aligned[i]
        camarilla_s4 = camarilla_s4_aligned[i]
        camarilla_pivot = camarilla_pivot_aligned[i]
        
        # --- 1d Trend Filter (using price vs Camarilla pivot) ---
        # Bullish 1d trend: price above pivot, Bearish: price below pivot
        trend_bullish = close[i] > camarilla_pivot
        trend_bearish = close[i] < camarilla_pivot
        
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
            
            # Exit conditions: trend reversal or opposite Camarilla level touch
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~24h)
            if min_hold:
                if position_side > 0:
                    # Exit long: trend turns bearish OR price touches S3 (mean reversion)
                    if trend_bearish or close[i] <= camarilla_s3:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: trend turns bullish OR price touches R3 (mean reversion)
                    if trend_bullish or close[i] >= camarilla_r3:
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
        # 1. Breakout above R4 with bullish 1d trend AND volume confirmation
        # 2. Mean reversion at S3 with weak/no 1d trend (price <= S3 and near pivot)
        if (bullish_breakout and trend_bullish and vol_ok) or \
           (close[i] <= camarilla_s3 and abs(close[i] - camarilla_pivot) < 0.5 * (camarilla_pivot - camarilla_s3) and not trend_bearish):
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # 1. Breakout below S4 with bearish 1d trend AND volume confirmation
        # 2. Mean reversion at R3 with weak/no 1d trend (price >= R3 and near pivot)
        elif (bearish_breakout and trend_bearish and vol_ok) or \
             (close[i] >= camarilla_r3 and abs(close[i] - camarilla_pivot) < 0.5 * (camarilla_r3 - camarilla_pivot) and not trend_bullish):
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals