#!/usr/bin/env python3
"""
Experiment #3211: 6h Donchian Breakout + 1d Weekly Pivot + Volume Spike
HYPOTHESIS: 6h Donchian(20) breakouts capture medium-term momentum with low trade frequency ideal for 6h timeframe.
1d weekly pivot levels (calculated from prior week's OHLC) provide institutional reference points: 
- Fade at R3/S3 levels in ranging markets
- Breakout continuation at R4/S4 levels in trending markets
Volume spike (>2.0x 20-period average) confirms breakout strength.
ATR-based trailing stop (2.5x) manages risk. Position size 0.25.
Designed to work in both bull (trend continuation from R4/S4) and bear (fade at R3/S3) markets.
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3211_6h_donchian20_1d_weekly_pivot_vol_v1"
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
    n_1d = len(close_1d)
    
    # Calculate weekly pivot points from prior week's OHLC
    # Week start: Monday 00:00 UTC
    # For each 1d bar, calculate pivot using prior week's (Mon-Sun) OHLC
    weekly_high = np.full(n_1d, np.nan)
    weekly_low = np.full(n_1d, np.nan)
    weekly_close = np.full(n_1d, np.nan)
    weekly_open = np.full(n_1d, np.nan)
    
    # Group 1d data by week (Mon-Sun)
    if n_1d >= 5:  # need at least a week
        # Create week labels for each 1d bar
        dates_1d = pd.to_datetime(df_1d.index)
        week_labels = dates_1d.isocalendar().year * 100 + dates_1d.isocalendar().week
        
        # Calculate weekly OHLC
        for week in np.unique(week_labels):
            mask = (week_labels == week)
            if np.sum(mask) > 0:
                week_idx = np.where(mask)[0]
                weekly_open[week_idx[-1]] = open_1d[week_idx[0]]  # Monday open
                weekly_high[week_idx[-1]] = np.max(high_1d[week_idx])
                weekly_low[week_idx[-1]] = np.min(low_1d[week_idx])
                weekly_close[week_idx[-1]] = close_1d[week_idx[-1]]  # Sunday close
    
    # Calculate pivot points: P = (H + L + C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    
    # Calculate support/resistance levels
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    # R4 = R3 + (H - L), S4 = S3 - (H - L)
    weekly_range = weekly_high - weekly_low
    r1 = 2 * weekly_pivot - weekly_low
    s1 = 2 * weekly_pivot - weekly_high
    r2 = weekly_pivot + weekly_range
    s2 = weekly_pivot - weekly_range
    r3 = weekly_high + 2 * (weekly_pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - weekly_pivot)
    r4 = r3 + weekly_range
    s4 = s3 - weekly_range
    
    # Align weekly pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
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
            np.isnan(pivot_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
            # Determine market regime based on price vs weekly pivot
            price_vs_pivot = price - pivot_aligned[i]
            
            # Long entry conditions:
            # 1. Breakout above Donchian high
            # 2. Either:
            #    a) Strong breakout above R4 (trend continuation) OR
            #    b) Fade from S3 in bearish extreme (mean reversion)
            donchian_breakout_long = price > highest_high[i]
            strong_breakout = price > r4_aligned[i]
            fade_from_support = price < s3_aligned[i] and price_vs_pivot < 0 and price > lowest_low[i] * 1.02  # near support
            
            if donchian_breakout_long and (strong_breakout or fade_from_support):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry conditions:
            # 1. Breakdown below Donchian low
            # 2. Either:
            #    a) Strong breakdown below S4 (trend continuation) OR
            #    b) Fade from R3 in bullish extreme (mean reversion)
            elif price < lowest_low[i]:
                strong_breakdown = price < s4_aligned[i]
                fade_from_resistance = price > r3_aligned[i] and price_vs_pivot > 0 and price < highest_high[i] * 0.98  # near resistance
                if strong_breakdown or fade_from_resistance:
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