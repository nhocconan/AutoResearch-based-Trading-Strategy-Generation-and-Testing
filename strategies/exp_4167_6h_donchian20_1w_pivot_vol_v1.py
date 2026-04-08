#!/usr/bin/env python3
"""
Experiment #4167: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation + ATR stoploss
HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot point bias (from prior week) capture momentum with reduced whipsaw. Volume spike (>2.0x) filters false breakouts. Weekly pivot provides structural bias that works in both bull (break above R1) and bear (break below S1) regimes. ATR trailing stop (2.5x) manages risk. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4167_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d for weekly pivot calculation (need daily data to build weekly) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 5:
        # Weekly pivot from prior week's OHLC
        # We'll use prior week's high, low, close to calculate pivot
        week_high = pd.Series(df_1d['high'].values).rolling(window=5, min_periods=5).max().values  # 5 trading days approx
        week_low = pd.Series(df_1d['low'].values).rolling(window=5, min_periods=5).min().values
        week_close = pd.Series(df_1d['close'].values).rolling(window=5, min_periods=5).last().values
        # Pivot point = (Prior Week High + Prior Week Low + Prior Week Close) / 3
        weekly_pivot = (week_high + week_low + week_close) / 3.0
        # Support/resistance levels
        r1 = 2 * weekly_pivot - week_low
        s1 = 2 * weekly_pivot - week_high
        r2 = weekly_pivot + (week_high - week_low)
        s2 = weekly_pivot - (week_high - week_low)
        r3 = week_high + 2 * (weekly_pivot - week_low)
        s3 = week_low - 2 * (week_high - weekly_pivot)
        # Align to 6h timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
        r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
        s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
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
    
    warmup = max(lookback_dc + 1, 20 + 5, 5 + 5, 14 + 5)  # DC lookback, vol MA buffer, weekly pivot buffer, ATR buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
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
        # Require volume spike (> 2.0x average) to filter noise
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Donchian breakout logic
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # Weekly pivot bias
            above_pivot = price > weekly_pivot_aligned[i]
            below_pivot = price < weekly_pivot_aligned[i]
            above_r1 = price > r1_aligned[i]
            below_s1 = price < s1_aligned[i]
            
            # Long conditions: Donchian breakout up + price above weekly pivot + above R1 (bullish bias)
            long_entry = breakout_up and above_pivot and above_r1
            
            # Short conditions: Donchian breakout down + price below weekly pivot + below S1 (bearish bias)
            short_entry = breakout_down and below_pivot and below_s1
            
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