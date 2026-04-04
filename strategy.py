#!/usr/bin/env python3
"""
Experiment #3271: 6h Donchian Breakout + 1d Weekly Pivot + Volume Confirmation
HYPOTHESIS: 6h Donchian(20) breakouts capture medium-term momentum with low trade frequency ideal for 6h timeframe.
1d weekly pivot (calculated from prior week's OHLC) provides structural bias: long above weekly pivot, short below.
Volume spike (>1.8x 20-period average) confirms breakout strength. ATR trailing stop (2.0x) manages risk.
Position size 0.25. Target: 75-150 total trades over 4 years (19-37/year).
Designed to work in bull markets (breakout continuation) and bear markets (mean reversion from extremes via pivot rejection).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3271_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    open_1d = df_1d['open'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior week's OHLC (using last 5 trading days)
    def calculate_weekly_pivot(dates, o, h, l, c):
        """Calculate weekly pivot from last 5 days OHLC"""
        n_days = len(o)
        if n_days < 5:
            return np.full(n_days, np.nan)
        
        weekly_pivot = np.full(n_days, np.nan)
        
        # For each day, use prior 5-day week's OHLC (shifted by 5)
        for i in range(5, n_days):
            # Prior week: i-5 to i-1 (5 days)
            week_o = o[i-5]  # Monday open
            week_h = np.max(h[i-5:i])  # Weekly high
            week_l = np.min(l[i-5:i])  # Weekly low
            week_c = c[i-1]  # Friday close
            
            # Weekly pivot = (High + Low + Close) / 3
            weekly_pivot[i] = (week_h + week_l + week_c) / 3.0
        
        return weekly_pivot
    
    weekly_pivot_1d = calculate_weekly_pivot(df_1d.index, open_1d, high_1d, low_1d, close_1d)
    weekly_pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot_1d)
    
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
    
    warmup = max(50, lookback, 20, 14, 5)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(weekly_pivot_1d_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
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
                # Exit if price rises 2.0*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
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
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Weekly pivot bias: long above pivot, short below pivot
            price_vs_pivot = price - weekly_pivot_1d_aligned[i]
            
            # Long entry: price breaks above Donchian high with bullish pivot bias
            if price > highest_high[i] and price_vs_pivot > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with bearish pivot bias
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