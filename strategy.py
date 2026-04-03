#!/usr/bin/env python3
"""
Experiment #115: 6h Donchian(20) Breakout + 1w Camarilla Pivot + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with weekly Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) capture high-probability moves.
Volume confirmation filters false breakouts. ATR-based trailing stop manages risk. Designed for 75-200 total trades over 4 years (19-50/year).
Works in bull/bear markets: in bull, breakouts at R4 continue; in bear, breakouts at S4 continue; in range, fades at R3/S3.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_camarilla_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    # Calculate weekly OHLC from 1w data
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    weekly_range = weekly_high - weekly_low
    
    # Camarilla levels: based on previous week's range
    # R4 = close + range * 1.1/2, R3 = close + range * 1.1/4, etc.
    camarilla_r4 = weekly_close + weekly_range * 1.1 / 2
    camarilla_r3 = weekly_close + weekly_range * 1.1 / 4
    camarilla_s3 = weekly_close - weekly_range * 1.1 / 4
    camarilla_s4 = weekly_close - weekly_range * 1.1 / 2
    
    # Align to 6h timeframe (shifted by 1 for completed weekly bars only)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
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
            np.isnan(vol_ma_20[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
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
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~18h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR breaks below S3 (mean reversion)
                    if close[i] <= dc_lower_20[i] or close[i] < camarilla_s3_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR breaks above R3 (mean reversion)
                    if close[i] >= dc_upper_20[i] or close[i] > camarilla_r3_aligned[i]:
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
        # 1. Breakout above upper Donchian with volume confirmation
        # 2. Either:
        #    a. Breakout continuation: price > R4 (strong momentum)
        #    b. Mean reversion: price < R3 but bouncing off support
        if bullish_breakout and vol_ok:
            if close[i] > camarilla_r4_aligned[i] or (close[i] < camarilla_r3_aligned[i] and close[i] > camarilla_s3_aligned[i]):
                in_position = True
                position_side = 1
                entry_bar = i
                highest_since_entry = high[i]
                signals[i] = SIZE
        # Short conditions:
        # 1. Breakout below lower Donchian with volume confirmation
        # 2. Either:
        #    a. Breakout continuation: price < S4 (strong momentum)
        #    b. Mean reversion: price > S3 but failing at resistance
        elif bearish_breakout and vol_ok:
            if close[i] < camarilla_s4_aligned[i] or (close[i] > camarilla_s3_aligned[i] and close[i] < camarilla_r3_aligned[i]):
                in_position = True
                position_side = -1
                entry_bar = i
                lowest_since_entry = low[i]
                signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals