#!/usr/bin/env python3
"""
Experiment #5681: 4h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: On 4h timeframe, Donchian(20) breakouts with volume > 1.8x average and aligned 
with 1d weekly pivot direction (price above weekly pivot = bullish bias, below = bearish bias) 
capture high-probability trend continuation moves. Weekly pivot provides structural support/resistance 
from higher timeframe that works in both bull and bear markets by avoiding entries against 
key weekly levels. Volume confirms breakout strength. ATR trailing stop (2.5x) manages risk. 
Discrete sizing (0.25) minimizes fee churn. Target: 19-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5681_4h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "4h"
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
        # Calculate weekly pivot from prior week's OHLC (using last 5 trading days approx)
        # We'll use rolling window of 5 days to get weekly high/low/close
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot point: (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Support/resistance levels
        r1 = 2 * weekly_pivot - weekly_low
        s1 = 2 * weekly_pivot - weekly_high
        r2 = weekly_pivot + (weekly_high - weekly_low)
        s2 = weekly_pivot - (weekly_high - weekly_low)
        r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
        s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
        r4 = weekly_high + 3 * (weekly_pivot - weekly_low)
        s4 = weekly_low - 3 * (weekly_high - weekly_pivot)
    else:
        # Fallback if insufficient data
        weekly_pivot = np.full(len(df_1d), np.nan)
        r1 = s1 = r2 = s2 = r3 = s3 = r4 = s4 = np.full(len(df_1d), np.nan)
    
    # Align weekly pivot and levels to 4h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
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
    
    warmup = max(20, 20, 14, 5)  # Donchian, volume avg, ATR, weekly pivot lookback
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below weekly S4 (strong bearish break)
                if price <= stop_price or price <= s4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above weekly R4 (strong bullish break)
                if price >= stop_price or price >= r4_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.8
        
        # Weekly pivot bias: long above weekly pivot, short below weekly pivot
        long_bias = price > weekly_pivot_aligned[i]
        short_bias = price < weekly_pivot_aligned[i]
        
        # Entry conditions: breakout in direction of weekly pivot bias with volume
        # Only take breakouts that align with weekly structure (avoid fakeouts)
        long_setup = breakout_up and volume_confirmed and long_bias and price > r1_aligned[i]
        short_setup = breakout_down and volume_confirmed and short_bias and price < s1_aligned[i]
        
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