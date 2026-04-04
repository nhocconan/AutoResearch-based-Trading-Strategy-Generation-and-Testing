#!/usr/bin/env python3
"""
Experiment #5879: 6h Donchian(20) breakout + 12h pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with 12h pivot levels (R1/S1, R2/S2) capture momentum with structure.
Volume confirmation filters weak breakouts. Uses 12h timeframe for pivot calculation to avoid look-ahead.
Designed for 6h timeframe to balance trade frequency (target: 75-200 total trades over 4 years).
Works in bull markets (breakouts above pivot resistance with volume) and bear markets
(breakdowns below pivot support with volume). ATR-based trailing stop manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5879_6h_donchian20_12h_pivot_vol_v1"
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
    
    # === HTF: 12h data for pivot points ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 2:
        high_12h = pd.Series(df_12h['high'])
        low_12h = pd.Series(df_12h['low'])
        close_12h = pd.Series(df_12h['close'])
        pivot = (high_12h + low_12h + close_12h) / 3
        r1 = 2 * pivot - low_12h
        s1 = 2 * pivot - high_12h
        r2 = pivot + (high_12h - low_12h)
        s2 = pivot - (high_12h - low_12h)
        # Align to 6h timeframe (shifted by 1 for completed bars only)
        pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot.values)
        r1_aligned = align_htf_to_ltf(prices, df_12h, r1.values)
        s1_aligned = align_htf_to_ltf(prices, df_12h, s1.values)
        r2_aligned = align_htf_to_ltf(prices, df_12h, r2.values)
        s2_aligned = align_htf_to_ltf(prices, df_12h, s2.values)
    else:
        pivot_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(20, 20, 20, 14)  # Donchian, volume avg, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below S1 (failed breakout)
                if price <= stop_price or price <= s1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above R1 (failed breakout)
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
        # Pivot direction filter: long above R1, short below S1
        pivot_long = price > r1_aligned[i]
        pivot_short = price < s1_aligned[i]
        
        # Entry conditions: breakout in direction of pivot with volume confirmation
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

</think>