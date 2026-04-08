#!/usr/bin/env python3
"""
Experiment #5787: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 1d weekly pivot levels (R1/S1) capture institutional order flow with volume confirmation. Weekly pivots act as dynamic support/resistance that work in both bull/bear markets by identifying key levels where smart money enters/exits. Targets 75-150 trades over 4 years with discrete sizing 0.25-0.30 to minimize fee drag. ATR-based trailing stop manages risk during volatile periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5787_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    
    # === HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Calculate weekly pivot from prior week's OHLC (using 5-day approximation for weekly)
        # Weekly high = max(high of last 5 days), weekly low = min(low of last 5 days), weekly close = close of 5th day ago
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(df_1d['close']).shift(4).values  # Close from 4 days ago (5th day back)
        weekly_open = pd.Series(df_1d['open']).shift(4).values   # Open from 4 days ago
        
        # Weekly pivot point calculation
        pp = (weekly_high + weekly_low + weekly_close) / 3.0
        r1 = 2 * pp - weekly_low
        s1 = 2 * pp - weekly_high
        r2 = pp + (weekly_high - weekly_low)
        s2 = pp - (weekly_high - weekly_low)
        r3 = weekly_high + 2 * (pp - weekly_low)
        s3 = weekly_low - 2 * (weekly_high - pp)
        r4 = pp + 3 * (weekly_high - weekly_low)
        s4 = pp - 3 * (weekly_high - weekly_low)
    else:
        # Not enough data for weekly calculation
        pp = r1 = s1 = r2 = s2 = r3 = s3 = r4 = s4 = np.full(len(df_1d), np.nan)
    
    # Align weekly pivot levels to 6h timeframe (shifted by 1 for completed 1d bars only)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = max(20, 20, 14, 5)  # Donchian, volume avg, ATR, weekly pivot warmup
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below S1 (failed hold above support)
                if price <= stop_price or price <= s1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above R1 (failed hold below resistance)
                if price >= stop_price or price >= r1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5
        # Weekly pivot filter: breakout in direction of R1/S1
        pivot_long = breakout_up and price > r1_aligned[i-1]  # Break above resistance
        pivot_short = breakout_down and price < s1_aligned[i-1]  # Break below support
        
        # Entry conditions: breakout with volume confirmation and weekly pivot alignment
        long_setup = breakout_up and volume_confirmed and pivot_long
        short_setup = breakout_down and volume_confirmed and pivot_short
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals