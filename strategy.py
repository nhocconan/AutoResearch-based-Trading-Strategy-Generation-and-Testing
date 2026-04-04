#!/usr/bin/env python3
"""
Experiment #5887: 6h Donchian(20) breakout + 1d weekly pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot levels capture strong momentum moves.
Weekly pivot (S1/S2/S3/R1/R2/R3) provides institutional support/resistance. Volume confirmation
filters weak breakouts. Works in bull markets (breakouts above weekly R1 with volume) and bear markets
(breakdowns below weekly S1 with volume). ATR-based trailing stop manages risk.
Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5887_6h_donchian20_1d_weekly_pivot_vol_v1"
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
        # Calculate weekly pivot from daily OHLC (using prior week's data)
        # We'll use rolling window of 5 days (1 week) to calculate pivot
        high_1d = pd.Series(df_1d['high'])
        low_1d = pd.Series(df_1d['low'])
        close_1d = pd.Series(df_1d['close'])
        
        # Weekly high/low/close (prior completed week)
        weekly_high = high_1d.rolling(window=5, min_periods=5).max().shift(1)  # shift(1) for prior week
        weekly_low = low_1d.rolling(window=5, min_periods=5).min().shift(1)
        weekly_close = close_1d.rolling(window=5, min_periods=5).mean().shift(1)
        
        # Weekly pivot point and support/resistance levels
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_r1 = 2 * weekly_pivot - weekly_low
        weekly_s1 = 2 * weekly_pivot - weekly_high
        weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
        weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
        weekly_r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
        weekly_s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
        
        # Align to 6h timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1.values)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1.values)
        weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2.values)
        weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2.values)
        weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3.values)
        weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3.values)
    else:
        # Fallback if insufficient data
        weekly_pivot_aligned = np.full(n, np.nan)
        weekly_r1_aligned = np.full(n, np.nan)
        weekly_s1_aligned = np.full(n, np.nan)
        weekly_r2_aligned = np.full(n, np.nan)
        weekly_s2_aligned = np.full(n, np.nan)
        weekly_r3_aligned = np.full(n, np.nan)
        weekly_s3_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 20, 5, 14)  # Donchian, volume avg, weekly pivot lookback, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below weekly S1 (failed breakout)
                if price <= stop_price or price <= weekly_s1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above weekly R1 (failed breakout)
                if price >= stop_price or price >= weekly_r1_aligned[i]:
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
        # Weekly pivot filter: long above weekly R1, short below weekly S1
        pivot_long = price > weekly_r1_aligned[i]
        pivot_short = price < weekly_s1_aligned[i]
        
        # Entry conditions: breakout in direction of weekly pivot with volume confirmation
        long_setup = breakout_up and pivot_long and volume_confirmed
        short_setup = breakout_down and pivot_short and volume_confirmed
        
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