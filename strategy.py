#!/usr/bin/env python3
"""
Experiment #335: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts on 6h timeframe, filtered by weekly pivot direction 
(price above weekly pivot = bullish bias, below = bearish bias) and 1d volume confirmation, 
creates a robust strategy that captures institutional breakouts in both bull and bear markets. 
Weekly pivot provides structural bias from higher timeframe, Donchian(20) captures 20-period 
breakouts with clear entry/exit levels, and 1d volume filter ensures participation. 
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    weekly_pivot = np.full(n, np.nan)
    weekly_r1 = np.full(n, np.nan)
    weekly_s1 = np.full(n, np.nan)
    weekly_r2 = np.full(n, np.nan)
    weekly_s2 = np.full(n, np.nan)
    
    # For each 6h bar, get the most recent completed weekly bar
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        # Find the most recent completed 1w bar before current 6h bar
        prior_weekly_bars = df_1w[df_1w['open_time'] < current_time]
        if len(prior_weekly_bars) > 0:
            prev_week = prior_weekly_bars.iloc[-1]
            ph = prev_week['high']
            pl = prev_week['low']
            pc = prev_week['close']
            
            # Standard pivot point formulas
            pivot = (ph + pl + pc) / 3.0
            weekly_pivot[i] = pivot
            weekly_r1[i] = 2 * pivot - pl
            weekly_s1[i] = 2 * pivot - ph
            weekly_r2[i] = pivot + (ph - pl)
            weekly_s2[i] = pivot - (ph - pl)
        else:
            # Not enough prior data
            weekly_pivot[i] = np.nan
            weekly_r1[i] = np.nan
            weekly_s1[i] = np.nan
            weekly_r2[i] = np.nan
            weekly_s2[i] = np.nan
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 6h Indicators ===
    # Calculate Donchian channel (20-period) on 6h data
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    if n >= lookback:
        # Calculate rolling max/min
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        highest_high[lookback-1:] = high_series.rolling(window=lookback, min_periods=lookback).max().values[lookback-1:]
        lowest_low[lookback-1:] = low_series.rolling(window=lookback, min_periods=lookback).min().values[lookback-1:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(lookback, 50)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(weekly_pivot[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Directional Bias: Weekly pivot determines long/short bias ---
        bullish_bias = close[i] > weekly_pivot[i]   # Price above weekly pivot = bullish
        bearish_bias = close[i] < weekly_pivot[i]   # Price below weekly pivot = bearish
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > highest_high[i]      # New 20-period high
        breakout_down = close[i] < lowest_low[i]      # New 20-period low
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using available data up to i
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Donchian band or weekly S1/R1
                if close[i] <= lowest_low[i] or close[i] >= weekly_r1[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Donchian band or weekly S1/R1
                if close[i] >= highest_high[i] or close[i] <= weekly_s1[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Bullish bias + Donchian breakout up + volume confirmation
        long_condition = bullish_bias and breakout_up and volume_spike
        
        # Short: Bearish bias + Donchian breakout down + volume confirmation
        short_condition = bearish_bias and breakout_down and volume_spike
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals