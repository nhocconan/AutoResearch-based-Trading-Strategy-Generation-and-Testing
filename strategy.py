#!/usr/bin/env python3
"""
Experiment #3815: 6h Donchian(20) breakout + 1w pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture medium-term swings. Weekly pivot direction (from 1w close vs prior 1w close) filters for trend alignment. Volume > 1.5x 20-period MA confirms institutional participation. Works in bull markets (breakouts above resistance in uptrend) and bear markets (breakdowns below support in downtrend). Discrete position sizing (0.25) minimizes fee drag. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3815_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for pivot direction (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w pivot direction: 1 = up (close > prior close), -1 = down (close < prior close)
    pivot_dir_1w = np.zeros(len(close_1w))
    for i in range(1, len(close_1w)):
        if close_1w[i] > close_1w[i-1]:
            pivot_dir_1w[i] = 1
        elif close_1w[i] < close_1w[i-1]:
            pivot_dir_1w[i] = -1
        else:
            pivot_dir_1w[i] = pivot_dir_1w[i-1]  # carry forward previous direction
    
    # Align 1w pivot direction to 6h timeframe (shifted by 1 for completed 1w bar)
    pivot_dir_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_dir_1w)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(pivot_dir_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (using Donchian width as proxy)
                dc_width = highest_high[i] - lowest_low[i]
                if dc_width > 0 and price < highest_since_entry - 2.0 * dc_width:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian lower band (trend reversal)
                elif price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (using Donchian width as proxy)
                dc_width = highest_high[i] - lowest_low[i]
                if dc_width > 0 and price > lowest_since_entry + 2.0 * dc_width:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian upper band (trend reversal)
                elif price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) AND pivot alignment
        volume_spike = vol_ratio[i] > 1.5
        pivot_up = pivot_dir_1w_aligned[i] > 0
        pivot_down = pivot_dir_1w_aligned[i] < 0
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND weekly pivot up
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                pivot_up):                     # Weekly trend is up
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND weekly pivot down
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  pivot_down):                   # Weekly trend is down
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