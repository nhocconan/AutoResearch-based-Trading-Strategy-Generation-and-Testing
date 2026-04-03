#!/usr/bin/env python3
"""
Experiment #351: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts on 6h timeframe, filtered by weekly pivot direction (from 1w HTF) 
and confirmed by 6h volume spike (>2.0x average), captures high-probability trend continuations. 
Weekly pivot provides structural bias (price above weekly pivot = bullish bias, below = bearish bias), 
reducing false breakouts in choppy markets. Volume confirmation ensures institutional participation. 
Targets 12-37 trades/year on 6h timeframe (50-150 total over 4 years) with discrete position sizing 
(0.25) to minimize fee drag while avoiding overtrading. Works in both bull (breakout continuation) 
and bear (breakdown continuation) markets by aligning with weekly structural bias.
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
    
    for i in range(n):
        current_time = prices.iloc[i]['open_time']
        # Find the most recent completed 1w bar before current 6h bar
        prior_1w_bars = df_1w[df_1w['open_time'] < current_time]
        if len(prior_1w_bars) > 0:
            prev_week = prior_1w_bars.iloc[-1]
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
            weekly_pivot[i] = np.nan
            weekly_r1[i] = np.nan
            weekly_s1[i] = np.nan
            weekly_r2[i] = np.nan
            weekly_s2[i] = np.nan
    
    # === HTF: 1d data for volume spike confirmation (Call ONCE before loop) ===
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
    # Donchian channel (20-period) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(lookback, 20) + 10  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(weekly_pivot[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Weekly Pivot Bias: Price above weekly pivot = bullish bias, below = bearish bias ---
        price_above_weekly_pivot = close[i] > weekly_pivot[i]
        price_below_weekly_pivot = close[i] < weekly_pivot[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > donchian_upper[i]  # Price breaks above upper band
        breakdown_dn = close[i] < donchian_lower[i]  # Price breaks below lower band
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
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
                # Take profit at Donchian middle (trailing profit protection)
                if close[i] <= donchian_middle[i]:
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
                # Take profit at Donchian middle (trailing profit protection)
                if close[i] >= donchian_middle[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up + weekly pivot bullish bias + volume spike
        long_condition = breakout_up and price_above_weekly_pivot and volume_spike
        
        # Short: Donchian breakdown down + weekly pivot bearish bias + volume spike
        short_condition = breakdown_dn and price_below_weekly_pivot and volume_spike
        
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