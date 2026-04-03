#!/usr/bin/env python3
"""
Experiment #420: 4h Donchian(20) Breakout + 1d Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, confirmed by 1d volume spike (>2.0x average), 
with ATR-based stoploss (2.5x ATR) and discrete position sizing (0.25) creates a robust strategy 
that works in both bull and bear markets. The Donchian structure captures institutional breakouts, 
volume confirms participation, and ATR stoploss manages risk. Targets 19-50 trades/year on 4h 
timeframe (75-200 total over 4 years) to minimize fee drag while capturing high-probability 
breakouts with strong volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_vol_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
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
    
    # === 4h Indicators: Donchian Channel (20) ===
    # Donchian upper/lower = highest high/lowest low of last 20 periods
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    if n >= 20:
        # Calculate rolling max/min for Donchian channels
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        donchian_upper[19:] = high_series.rolling(window=20, min_periods=20).max().values[19:]
        donchian_lower[19:] = low_series.rolling(window=20, min_periods=20).min().values[19:]
        # For warmup period, use expanding window
        for i in range(20):
            donchian_upper[i] = high[:i+1].max()
            donchian_lower[i] = low[:i+1].min()
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0  # For trailing stop logic
    lowest_since_entry = 0.0
    
    warmup = 50  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                # Update highest price since entry
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.5 * ATR below highest since entry
                stop_level = highest_since_entry - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian lower (contrarian exit) or time-based
                if close[i] <= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Update lowest price since entry
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.5 * ATR above lowest since entry
                stop_level = lowest_since_entry + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian upper (contrarian exit)
                if close[i] >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: require significant volume spike (> 2.0x average)
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # Long: Price breaks above Donchian upper with volume confirmation
        long_condition = (
            close[i] > donchian_upper[i] and 
            volume_spike
        )
        
        # Short: Price breaks below Donchian lower with volume confirmation
        short_condition = (
            close[i] < donchian_lower[i] and 
            volume_spike
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals