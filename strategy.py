#!/usr/bin/env python3
"""
Experiment #3459: 6h Donchian Breakout + 12h Weekly Pivot Direction + Volume Spike
HYPOTHESIS: 6h Donchian(20) breakouts capture medium-term trends with ideal trade frequency for 6h timeframe.
12h weekly pivot (from 1d data) provides institutional reference levels for direction bias.
Volume spike (>2.0x 20-period average) confirms breakout strength. Works in bull markets via trend continuation
and bear markets via mean reversion from extreme pivot levels (R4/S4) when price overextends.
Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity with fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3459_6h_donchian20_12h_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for weekly pivot points (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate weekly pivot points from prior week (12h data ≈ 1.5 days per bar)
    # Need ~7 bars for 1 week (7 * 12h = 3.5 days, use 14 bars for safety)
    lookback_weeks = 14  # ~2 weeks of 12h data for prior week calculation
    if len(high_12h) >= lookback_weeks:
        # Rolling window for prior week's high/low/close
        prev_week_high = pd.Series(high_12h).rolling(window=lookback_weeks, min_periods=lookback_weeks).max().shift(1).values
        prev_week_low = pd.Series(low_12h).rolling(window=lookback_weeks, min_periods=lookback_weeks).min().shift(1).values
        prev_week_close = pd.Series(close_12h).rolling(window=lookback_weeks, min_periods=lookback_weeks).mean().shift(1).values
        
        # Weekly pivot point calculation
        pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
        width = prev_week_high - prev_week_low
        
        # Camarilla-style weekly levels
        r4 = pivot + (width * 1.1 / 2)  # R4
        r3 = pivot + (width * 1.1 / 4)  # R3
        s3 = pivot - (width * 1.1 / 4)  # S3
        s4 = pivot - (width * 1.1 / 2)  # S4
        
        # Align to 6h timeframe
        pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
        r4_aligned = align_htf_to_ltf(prices, df_12h, r4)
        r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
        s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
        s4_aligned = align_htf_to_ltf(prices, df_12h, s4)
    else:
        # Not enough data yet
        pivot_aligned = r4_aligned = r3_aligned = s3_aligned = s4_aligned = np.full(n, np.nan)
    
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
    
    warmup = max(50, lookback, 20, 14, lookback_weeks)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
            # Weekly pivot direction bias: long above pivot, short below pivot
            price_vs_pivot = price - pivot_aligned[i]
            
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