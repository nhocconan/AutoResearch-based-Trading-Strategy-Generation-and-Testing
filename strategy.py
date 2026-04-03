#!/usr/bin/env python3
"""
Experiment #099: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot levels (R4/S4 breakout, R3/S3 fade)
capture institutional order flow with clear structure. Weekly pivots from prior week provide
key support/resistance that price reacts to. Volume confirmation ensures breakout legitimacy.
Works in bull/bear markets by trading breakouts in direction of weekly bias (above/below weekly pivot).
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate weekly pivot from prior week's OHLC (using 1d data)
    # We need to group 1d data into weeks, but for simplicity, we'll use rolling window of 5 days
    # and calculate pivot points from the prior week's high/low/close
    # Since we don't have explicit week grouping, we approximate with 5-day lookback
    if len(df_1d) >= 5:
        # Get prior week's (5-day ago) OHLC for weekly pivot calculation
        # We'll use the 5-day period ending 1 day ago (yesterday's 5-day window)
        # This ensures we only use completed weekly data
        week_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1).values
        week_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1).values
        week_close = df_1d['close'].rolling(window=5, min_periods=5).mean().shift(1).values
        
        # Weekly pivot points (standard calculation)
        pw = (week_high + week_low + week_close) / 3.0
        r1 = 2 * pw - week_low
        s1 = 2 * pw - week_high
        r2 = pw + (week_high - week_low)
        s2 = pw - (week_high - week_low)
        r3 = week_high + 2 * (pw - week_low)
        s3 = week_low - 2 * (week_high - pw)
        r4 = week_high + 3 * (pw - week_low)
        s4 = week_low - 3 * (week_high - pw)
        
        # Align to 6h timeframe
        pw_aligned = align_htf_to_ltf(prices, df_1d, pw)
        r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
        s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    else:
        # Not enough data for weekly pivot
        pw_aligned = np.full(n, np.nan)
        r4_aligned = np.full(n, np.nan)
        s4_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # ATR for volatility and stoploss
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian channels (20-period)
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (1.5x 20-period average)
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
            np.isnan(vol_ma_20[i]) or np.isnan(pw_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Levels ---
        # Bias: above weekly pivot = bullish, below = bearish
        weekly_bullish = close[i] > pw_aligned[i]
        weekly_bearish = close[i] < pw_aligned[i]
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False
        
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
            
            # Exit conditions: 
            # 1. Price reaches opposite weekly pivot level (R4/S4) - take profit
            # 2. Price reverses at weekly support/resistance (R3/S3) - fade
            # 3. Trend reversal (close crosses weekly pivot)
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: 
                    # - Take profit at R4
                    # - Fade at R3 if price fails to break through
                    # - Stop if price crosses below weekly pivot
                    if close[i] >= r4_aligned[i] or \
                       (close[i] <= r3_aligned[i] and close[i] > s3_aligned[i]) or \
                       close[i] < pw_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short:
                    # - Take profit at S4
                    # - Fade at S3 if price fails to break down
                    # - Stop if price crosses above weekly pivot
                    if close[i] <= s4_aligned[i] or \
                       (close[i] >= s3_aligned[i] and close[i] < r3_aligned[i]) or \
                       close[i] > pw_aligned[i]:
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
        # Breakout above upper Donchian with bullish weekly bias and volume confirmation
        # Additional filter: avoid buying into strong resistance (below R4)
        if bullish_breakout and weekly_bullish and vol_ok and close[i] < r4_aligned[i]:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with bearish weekly bias and volume confirmation
        # Additional filter: avoid selling into strong support (above S4)
        elif bearish_breakout and weekly_bearish and vol_ok and close[i] > s4_aligned[i]:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals