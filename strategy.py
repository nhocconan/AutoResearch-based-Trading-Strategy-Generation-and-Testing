#!/usr/bin/env python3
"""
Experiment #3127: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture medium-term trends with controlled frequency. 
1d weekly pivot (based on prior week's H/L/C) provides directional bias: long only above weekly pivot, short only below. 
Volume spike (>2.0x 20-period average) confirms breakout strength. ATR trailing stop (2.5x) manages risk. 
Position size 0.25. Target: 75-200 total trades over 4 years (19-50/year). 
Weekly pivot filter reduces false breakouts in ranging markets, improving performance in both bull (trend continuation) 
and bear (mean reversion from extremes) by aligning with higher timeframe structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3127_6h_donchian20_1d_pivot_vol_v1"
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior week's OHLC
    # Weekly pivot = (Prior Week High + Prior Week Low + Prior Week Close) / 3
    # We need to shift by 1 week to avoid look-ahead (use prior completed week)
    # Since we have daily data, we approximate weekly by taking last 5 days
    # But simpler: use prior day's values as proxy for weekly bias (more responsive)
    # Actually, let's compute true weekly pivot using 1d data grouped by week
    # However, to avoid complexity and look-ahead, we use: 
    # Weekly Pivot = (yesterday's high + yesterday's low + yesterday's close) / 3
    # This is actually the standard daily pivot, but we'll call it "weekly pivot" for HTF bias
    # To get prior week's values, we need to look back 5 trading days
    lookback_week = 5
    if len(high_1d) >= lookback_week:
        # Rolling window of 5 days to get weekly high/low/close
        weekly_high = pd.Series(high_1d).rolling(window=lookback_week, min_periods=lookback_week).max().shift(1).values
        weekly_low = pd.Series(low_1d).rolling(window=lookback_week, min_periods=lookback_week).min().shift(1).values
        weekly_close = pd.Series(close_1d).rolling(window=lookback_week, min_periods=lookback_week).last().shift(1).values
        # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    else:
        weekly_pivot = np.full_like(close_1d, np.nan)
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price <= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price re-enters Donchian channel (mean reversion)
                elif price >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Weekly pivot direction filter: only long above pivot, short below pivot
            price_vs_pivot = price - weekly_pivot_aligned[i]
            
            # Long entry: price breaks above Donchian high AND above weekly pivot
            if price > highest_high[i] and price_vs_pivot > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND below weekly pivot
            elif price < lowest_low[i] and price_vs_pivot < 0:
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