#!/usr/bin/env python3
"""
Experiment #3079: 6h Donchian(20) breakout + 12h volume spike + 1d pivot regime
HYPOTHESIS: 6h Donchian breakouts capture medium-term trends with controlled frequency. 
Volume spike (>2.0x 20-period average on 12h) confirms breakout strength. 
1d Camarilla pivot levels filter direction: only long above R4, short below S4. 
This combination should work in both bull (breakout continuation) and bear (mean reversion from extremes) 
markets by using price channels and volatility filters with strict pivot-based regime filtering.
Target: 75-200 total trades over 4 years (19-50/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3079_6h_donchian20_12h_vol_1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike confirmation (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate volume MA(20) on 12h for spike detection
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = np.ones(len(volume_12h))
    vol_ratio_12h[20:] = volume_12h[20:] / vol_ma_12h[20:]
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Camarilla: R4 = close + ((high - low) * 1.1/2), S4 = close - ((high - low) * 1.1/2)
    # Using prior day's OHLC for current day's levels
    pivot_high = high_1d
    pivot_low = low_1d
    pivot_close = close_1d
    camarilla_r4 = pivot_close + ((pivot_high - pivot_low) * 1.1 / 2)
    camarilla_s4 = pivot_close - ((pivot_high - pivot_low) * 1.1 / 2)
    
    # Align pivot levels to 6h timeframe (using prior completed 1d bar)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Indicators: Donchian channels (20-period) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (using 6h ATR)
                tr = np.maximum(high[i] - low[i], 
                               np.abs(high[i] - close[i-1]), 
                               np.abs(low[i] - close[i-1]))
                # Simplified ATR approximation for exit - use 2.5% of price as proxy
                atr_proxy = price * 0.025
                if price < highest_since_entry - 2.5 * atr_proxy:
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
                tr = np.maximum(high[i] - low[i], 
                               np.abs(high[i] - close[i-1]), 
                               np.abs(low[i] - close[i-1]))
                atr_proxy = price * 0.025
                if price > lowest_since_entry + 2.5 * atr_proxy:
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
        # Require volume spike (> 2.0x average) on 12h for confirmation
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
        if volume_spike:
            # 1d Camarilla pivot regime filter: only long above R4, short below S4
            price_vs_r4 = price - camarilla_r4_aligned[i]
            price_vs_s4 = price - camarilla_s4_aligned[i]
            
            # Long entry: price breaks above Donchian high with price above R4 pivot
            if price > highest_high[i] and price_vs_r4 > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low with price below S4 pivot
            elif price < lowest_low[i] and price_vs_s4 < 0:
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