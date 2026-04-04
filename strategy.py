#!/usr/bin/env python3
"""
Experiment #4847: 6h Donchian(20) Breakout + 1d Weekly Pivot + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (price above/below weekly pivot) with volume confirmation (>1.5x average) capture strong momentum moves. Weekly pivot provides structural support/resistance from higher timeframe. Designed for 12-37 trades/year on 6h timeframe to minimize fee drag while maintaining statistical significance. Works in bull markets (breakouts above weekly pivot) and bear markets (breakdowns below weekly pivot).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4847_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1d data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # === 1d Indicators: Weekly Pivot (using last complete week's OHLC) ===
    # Calculate weekly OHLC from daily data
    if len(df_1d) >= 5:
        # Group into weeks (5 trading days approx)
        weeks = len(df_1d) // 5
        if weeks > 0:
            # Use last complete week's data
            week_start = (weeks - 1) * 5
            week_end = week_start + 5
            week_data = df_1d.iloc[week_start:week_end]
            
            if len(week_data) >= 5:
                week_high = week_data['high'].max()
                week_low = week_data['low'].min()
                week_close = week_data['close'].iloc[-1]
                
                # Standard pivot calculation
                weekly_pivot = (week_high + week_low + week_close) / 3.0
                weekly_r1 = 2 * weekly_pivot - week_low
                weekly_s1 = 2 * weekly_pivot - week_high
                weekly_r2 = weekly_pivot + (week_high - week_low)
                weekly_s2 = weekly_pivot - (week_high - week_low)
                weekly_r3 = weekly_high + 2 * (weekly_pivot - week_low)
                weekly_s3 = weekly_low - 2 * (week_high - weekly_pivot)
            else:
                weekly_pivot = weekly_r1 = weekly_s1 = weekly_r2 = weekly_s2 = weekly_r3 = weekly_s3 = np.nan
        else:
            weekly_pivot = weekly_r1 = weekly_s1 = weekly_r2 = weekly_s2 = weekly_r3 = weekly_s3 = np.nan
    else:
        weekly_pivot = weekly_r1 = weekly_s1 = weekly_r2 = weekly_s2 = weekly_r3 = weekly_s3 = np.nan
    
    # Align HTF weekly pivot to 6h timeframe
    if not np.isnan(weekly_pivot):
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), weekly_pivot))
        weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), weekly_r3))
        weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, np.full(len(df_1d), weekly_s3))
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
        weekly_r3_aligned = np.full(n, np.nan)
        weekly_s3_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (1.5x spike) ===
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
    
    warmup = max(20, 20, 14)  # Donchian, Volume MA, ATR warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or 
            np.isnan(weekly_s3_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions with weekly pivot alignment
        breakout_long = (price >= high_roll[i]) and (price > weekly_pivot_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < weekly_pivot_aligned[i]) and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals