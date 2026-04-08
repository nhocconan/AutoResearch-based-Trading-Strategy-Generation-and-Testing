#!/usr/bin/env python3
"""
Experiment #4117: 4h Donchian(20) breakout + 1d EMA200 trend filter + volume confirmation
HYPOTHESIS: In both bull and bear markets, 4h breakouts aligned with the 1d EMA200 (long-term trend) and confirmed by volume spikes capture sustained moves while avoiding whipsaws. The 1d EMA200 acts as dynamic support/resistance. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4117_4h_donchian20_1d_ema200_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d EMA200 for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 200:
        ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, min_periods=200, adjust=False).mean().values
        ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    else:
        ema200_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel(20) for breakout levels ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
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
    
    warmup = max(lookback_dc + 1, 20 + 10)  # DC lookback, vol MA buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema200_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                # We'll use price action stop instead of ATR for simplicity in 4h
                if price < highest_since_entry - 0.05 * price:  # 5% trailing stop
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 0.05 * price:  # 5% trailing stop
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) to filter noise
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Breakout levels
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # 1d EMA200 trend filter
            above_ema200 = price > ema200_1d_aligned[i]
            below_ema200 = price < ema200_1d_aligned[i]
            
            # Long: breakout above Donchian high + above 1d EMA200
            long_entry = breakout_up and above_ema200
            # Short: breakout below Donchian low + below 1d EMA200
            short_entry = breakout_down and below_ema200
            
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