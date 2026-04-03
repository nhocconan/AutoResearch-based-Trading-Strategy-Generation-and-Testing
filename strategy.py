#!/usr/bin/env python3
"""
Experiment #274: 1h Donchian(20) breakout + 4h/1d pivot direction + volume confirmation
HYPOTHESIS: 1h Donchian breakouts aligned with 4h/1d weekly pivot levels capture intraday momentum in both bull/bear markets. Using 4h/1d for signal direction reduces false breakouts. Volume filter (>1.5x average) ensures participation. Session filter (08-20 UTC) avoids low-liquidity hours. Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_274_1h_donchian20_4h_1d_pivot_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 4h and 1d data (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # === 4h: Weekly pivot points from prior week ===
    week_high_4h = df_4h['high'].rolling(window=5, min_periods=5).max().shift(1)
    week_low_4h = df_4h['low'].rolling(window=5, min_periods=5).min().shift(1)
    week_close_4h = df_4h['close'].rolling(window=5, min_periods=5).last().shift(1)
    
    pivot_4h = (week_high_4h + week_low_4h + week_close_4h) / 3.0
    r1_4h = 2 * pivot_4h - week_low_4h
    s1_4h = 2 * pivot_4h - week_high_4h
    
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h.values)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h.values)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h.values)
    
    # === 1d: Weekly pivot points from prior week ===
    week_high_1d = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1)
    week_low_1d = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1)
    week_close_1d = df_1d['close'].rolling(window=5, min_periods=5).last().shift(1)
    
    pivot_1d = (week_high_1d + week_low_1d + week_close_1d) / 3.0
    r1_1d = 2 * pivot_1d - week_low_1d
    s1_1d = 2 * pivot_1d - week_high_1d
    
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d.values)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d.values)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d.values)
    
    # === 1h: Donchian(20) channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1h: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1h: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Session filter: 08-20 UTC ===
    # prices.index is DatetimeIndex, .hour works directly
    hours = prices.index.hour
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # Enough for 20-period indicators and 5-day weekly pivot
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(pivot_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or
            np.isnan(s1_4h_aligned[i]) or np.isnan(pivot_1d_aligned[i]) or
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # --- Combined Pivot Direction Logic (4h AND 1d agreement) ---
        # Long bias: price above R1 on BOTH 4h and 1d (strong bullish)
        # Short bias: price below S1 on BOTH 4h and 1d (strong bearish)
        long_bias = (price > r1_4h_aligned[i]) and (price > r1_1d_aligned[i])
        short_bias = (price < s1_4h_aligned[i]) and (price < s1_1d_aligned[i])
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on breakout down with volume if bearish bias on either TF
                if breakout_down and volume_spike and ((price < s1_4h_aligned[i]) or (price < s1_1d_aligned[i])):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on breakout up with volume if bullish bias on either TF
                if breakout_up and volume_spike and ((price > r1_4h_aligned[i]) or (price > r1_1d_aligned[i])):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require volume spike + breakout conditions + pivot bias alignment on BOTH timeframes
        if volume_spike:
            # Long: breakout up AND bullish bias on BOTH 4h and 1d
            if breakout_up and long_bias:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout down AND bearish bias on BOTH 4h and 1d
            elif breakout_down and short_bias:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals