#!/usr/bin/env python3
"""
Experiment #5995: 6h Camarilla pivot levels from 1d + volume confirmation
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) derived from prior 1d candle provide intraday structure. 
At 6h timeframe, we fade touches of R3/S3 with volume confirmation for mean reversion in ranging markets, 
and break R4/S4 with volume for continuation in trending markets. Weekly 1d pivot bias filters direction. 
Works in both bull/bear: mean reversion in chop, breakout in trend, volume avoids false signals.
Target: 75-150 trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5995_6h_camarilla1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for Camarilla levels and weekly pivot bias ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Prior 1d OHLC for Camarilla calculation
        prev_close = df_1d['close'].shift(1).values
        prev_high = df_1d['high'].shift(1).values
        prev_low = df_1d['low'].shift(1).values
        prev_range = prev_high - prev_low
        
        # Camarilla levels (based on prior day)
        camarilla_h5 = prev_close + 1.1 * prev_range / 6  # R4
        camarilla_h4 = prev_close + 1.1 * prev_range / 4  # R3
        camarilla_h3 = prev_close + 1.1 * prev_range / 3  # S3 equivalent (but we'll use as upper mean rev)
        camarilla_l3 = prev_close - 1.1 * prev_range / 3  # S3 equivalent
        camarilla_l4 = prev_close - 1.1 * prev_range / 4  # S4
        camarilla_l5 = prev_close - 1.1 * prev_range / 6  # R4 equivalent (lower)
        
        # Weekly pivot bias from prior week's 1d OHLC
        if len(df_1d) >= 5:
            weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
            weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
            weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1).values
            weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        else:
            weekly_pivot = np.full(len(df_1d), np.nan)
        
        # Align all HTF values to 6h
        camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
        camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
        camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
        camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
        camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
        camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    else:
        camarilla_h5_aligned = camarilla_h4_aligned = camarilla_h3_aligned = camarilla_l3_aligned = camarilla_l4_aligned = camarilla_l5_aligned = weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 6h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 14, 5) + 1  # Volume avg, ATR, weekly lookback + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price fails to hold above camarilla_l3 (for mean reversion) 
                # OR breaks camarilla_h5 (for breakout continuation - we take profit)
                if price <= stop_price or price <= camarilla_l3_aligned[i] or price >= camarilla_h5_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price fails to hold below camarilla_h3 (for mean reversion)
                # OR breaks camarilla_l5 (for breakout continuation - we take profit)
                if price >= stop_price or price >= camarilla_h3_aligned[i] or price <= camarilla_l5_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        volume_confirmed = volume_ratio[i] > 1.5
        
        # Determine market regime via weekly pivot bias
        above_weekly_pivot = price > weekly_pivot_aligned[i]
        below_weekly_pivot = price < weekly_pivot_aligned[i]
        
        # Mean reversion entries (fade extremes) - work in ranging markets
        mr_long = (price <= camarilla_l4_aligned[i]) and volume_confirmed and above_weekly_pivot
        mr_short = (price >= camarilla_h4_aligned[i]) and volume_confirmed and below_weekly_pivot
        
        # Breakout continuation entries - work in trending markets
        breakout_long = (price >= camarilla_h5_aligned[i]) and volume_confirmed and above_weekly_pivot
        breakout_short = (price <= camarilla_l5_aligned[i]) and volume_confirmed and below_weekly_pivot
        
        if mr_long or breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif mr_short or breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals