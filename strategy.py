#!/usr/bin/env python3
"""
Experiment #5719: 6h Donchian(20) breakout + 12h volume confirmation + 1d pivot direction filter
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts with volume > 2.0x average and aligned with 
daily pivot direction (price above daily pivot = bullish, below = bearish) capture high-probability 
trend continuation moves. The 12h volume filter ensures breakout strength while avoiding low-volume 
fakeouts. Daily pivot provides structural support/resistance that works in both bull and bear markets. 
Discrete sizing (0.25) minimizes fee churn. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5719_6h_donchian20_12h_vol_1d_pivot_v1"
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
    
    # === HTF: 12h data for volume confirmation ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        avg_volume_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    else:
        avg_volume_12h = np.full(len(df_12h), np.nan)
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    
    # === HTF: 1d data for pivot points (using prior day's OHLC) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Calculate pivot points from prior day's OHLC
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        pivot = (high_1d + low_1d + close_1d) / 3.0
        r1 = 2 * pivot - low_1d
        s1 = 2 * pivot - high_1d
        r2 = pivot + (high_1d - low_1d)
        s2 = pivot - (high_1d - low_1d)
        r3 = high_1d + 2 * (pivot - low_1d)
        s3 = low_1d - 2 * (high_1d - pivot)
    else:
        pivot = r1 = s1 = r2 = s2 = r3 = s3 = np.full(len(df_1d), np.nan)
    
    # Align pivot levels to 6h timeframe (shifted by 1 for completed daily bars only)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 6h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = max(20, 20, 14)  # Donchian, volume avg, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume[i]) or np.isnan(volume_12h_aligned[i]) or
            np.isnan(atr[i]) or np.isnan(pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below daily pivot (trend change)
                if price <= stop_price or price <= pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above daily pivot (trend change)
                if price >= stop_price or price >= pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume[i] > 2.0 * volume_12h_aligned[i]
        
        # Daily pivot bias: long above pivot, short below pivot
        long_bias = price > pivot_aligned[i]
        short_bias = price < pivot_aligned[i]
        
        # Entry conditions: breakout in direction of daily pivot with volume
        long_setup = breakout_up and volume_confirmed and long_bias
        short_setup = breakout_down and volume_confirmed and short_bias
        
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