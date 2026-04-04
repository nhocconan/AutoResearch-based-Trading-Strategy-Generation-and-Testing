#!/usr/bin/env python3
"""
Experiment #4179: 6h Donchian(20) breakout + 12h pivot direction + volume confirmation
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts provide high-probability trend continuation 
when aligned with 12h pivot levels (price > R1 for long, price < S1 for short) and confirmed 
by volume spikes (>1.5x average). Uses 0.25 position size to limit drawdown and targets 
100-200 total trades over 4 years (25-50/year). Works in both bull/bear via pivot direction filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4179_6h_donchian20_12h_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 12h data for pivot points ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 20:
        # Calculate daily pivot points from previous 12h bar
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # Pivot point calculation: PP = (H + L + C) / 3
        pp = (high_12h + low_12h + close_12h) / 3.0
        # Resistance and Support levels
        r1 = 2 * pp - low_12h
        s1 = 2 * pp - high_12h
        r2 = pp + (high_12h - low_12h)
        s2 = pp - (high_12h - low_12h)
        r3 = high_12h + 2 * (pp - low_12h)
        s3 = low_12h - 2 * (high_12h - pp)
        
        # Align to 6h timeframe (shifted by 1 for completed bars only)
        pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
        r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
        s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
        r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
        s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
        r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
        s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    else:
        pp_aligned = np.full(n, np.nan)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
        r3_aligned = np.full(n, np.nan)
        s3_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high, low, 20)
    
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
    
    warmup = max(20 + 5, 20 + 5, 14 + 5)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.0*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.0*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Donchian breakout conditions
            breakout_up = price > donchian_upper[i]
            breakout_down = price < donchian_lower[i]
            
            # Pivot direction filters: price > R1 = bullish bias, price < S1 = bearish bias
            bullish_pivot = price > r1_aligned[i]
            bearish_pivot = price < s1_aligned[i]
            
            # Long conditions: Donchian breakout up + bullish pivot + volume spike
            long_entry = breakout_up and bullish_pivot
            
            # Short conditions: Donchian breakout down + bearish pivot + volume spike
            short_entry = breakout_down and bearish_pivot
            
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