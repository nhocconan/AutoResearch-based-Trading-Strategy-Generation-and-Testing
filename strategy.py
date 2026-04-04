#!/usr/bin/env python3
"""
Experiment #4347: 6h Donchian Breakout + Weekly Pivot Direction + Volume Spike
HYPOTHESIS: 6h Donchian(20) breakouts in direction of weekly pivot trend (price > weekly pivot = long bias, price < weekly pivot = short bias) with volume confirmation (>2.0x average) capture institutional breakout attempts aligned with higher-timeframe structure. Weekly pivot provides objective trend filter reducing whipsaw in ranging markets. Targets 75-150 total trades over 4 years (19-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4347_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1d and 1w data for pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # === Calculate Weekly Pivot Points from 1w data ===
    # Weekly Pivot = (High + Low + Close) / 3
    # R1 = 2*P - Low, S1 = 2*P - High
    # R2 = P + (High - Low), S2 = P - (High - Low)
    # R3 = High + 2*(P - Low), S3 = Low - 2*(High - P)
    
    if len(df_1w) >= 1:
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values
        weekly_close = df_1w['close'].values
        
        # Calculate pivot points
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_range = weekly_high - weekly_low
        
        # Support and Resistance levels
        r1 = 2 * weekly_pivot - weekly_low
        s1 = 2 * weekly_pivot - weekly_high
        r2 = weekly_pivot + weekly_range
        s2 = weekly_pivot - weekly_range
        r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
        s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
        
        # Align to 6h timeframe (shifted by 1 for completed weekly bars only)
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
        r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    lookback = 20
    donchian_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(weekly_pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        if volume_confirm:
            # Determine bias from weekly pivot: price above pivot = long bias, below = short bias
            # Only take breakouts in direction of weekly pivot bias
            price_above_pivot = price > weekly_pivot_aligned[i]
            price_below_pivot = price < weekly_pivot_aligned[i]
            
            # Long conditions: Donchian upper breakout + price above weekly pivot + volume
            long_entry = (price > donchian_upper[i]) and price_above_pivot
            
            # Short conditions: Donchian lower breakout + price below weekly pivot + volume
            short_entry = (price < donchian_lower[i]) and price_below_pivot
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals