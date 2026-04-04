#!/usr/bin/env python3
"""
Experiment #5255: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (from 1w timeframe) provide high-probability trend entries. Enter long when price breaks above Donchian(20) high AND weekly pivot > prior weekly pivot (bullish bias) with volume confirmation (>1.5x). Enter short when price breaks below Donchian(20) low AND weekly pivot < prior weekly pivot (bearish bias) with volume confirmation. Designed for 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag. Weekly pivot filter ensures we only trade with the higher-timeframe trend, reducing whipsaw in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5255_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute HTF: 1w data for weekly pivot direction
    df_1w = get_htf_data(prices, '1w')
    
    # === 1w Indicators: Weekly Pivot Direction (using prior week's OHLC) ===
    if len(df_1w) >= 1:
        # Prior week OHLC (shifted by 1 to avoid look-ahead)
        prior_week_high = df_1w['high'].shift(1).values
        prior_week_low = df_1w['low'].shift(1).values
        prior_week_close = df_1w['close'].shift(1).values
        
        # Weekly pivot point
        weekly_pivot = (prior_week_high + prior_week_low + prior_week_close) / 3.0
        # Prior week's pivot for direction
        prior_weekly_pivot = np.roll(weekly_pivot, 1)
        prior_weekly_pivot[0] = np.nan
        
        # Weekly pivot direction: 1 = bullish (pivot rising), -1 = bearish (pivot falling)
        weekly_pivot_dir = np.where(weekly_pivot > prior_weekly_pivot, 1, 
                                   np.where(weekly_pivot < prior_weekly_pivot, -1, 0))
        
        # Align to 6h timeframe (shift(1) in align_htf_to_ltf ensures prior week only)
        weekly_pivot_dir_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_dir.astype(np.float64))
    else:
        weekly_pivot_dir_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_dir_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume filter: confirmation (>1.5x)
        vol_confirm = vol_ratio[i] > 1.5
        
        # Donchian breakout conditions
        breakout_long = price > donchian_high[i]
        breakout_short = price < donchian_low[i]
        
        # Weekly pivot direction filter
        pivot_bullish = weekly_pivot_dir_aligned[i] > 0
        pivot_bearish = weekly_pivot_dir_aligned[i] < 0
        
        # Final entry conditions: Donchian breakout + weekly pivot direction + volume
        if breakout_long and pivot_bullish and vol_confirm:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short and pivot_bearish and vol_confirm:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals