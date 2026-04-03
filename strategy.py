#!/usr/bin/env python3
"""
Experiment #699: 6h Donchian(20) breakout + 12h pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts filtered by 12h Camarilla pivot direction (R3/S3 for fading, R4/S4 for breakout) 
and volume confirmation captures institutional order flow with proper structure. Works in bull/bear markets via 
pivot-based regime filter: long when price > daily pivot and breaking R4, short when price < daily pivot and breaking S4. 
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_699_6h_donchian20_12h_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Camarilla pivot levels (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for 12h timeframe (using previous bar's data)
    pivot_12h = np.zeros_like(close_12h)
    r3_12h = np.zeros_like(close_12h)
    s3_12h = np.zeros_like(close_12h)
    r4_12h = np.zeros_like(close_12h)
    s4_12h = np.zeros_like(close_12h)
    
    for i in range(1, len(close_12h)):
        # Use previous bar's high/low/close for today's pivot levels
        ph = high_12h[i-1]
        pl = low_12h[i-1]
        pc = close_12h[i-1]
        
        pivot_12h[i] = (ph + pl + pc) / 3
        range_12h = ph - pl
        
        # Camarilla levels
        r3_12h[i] = pc + range_12h * 1.1 / 4
        s3_12h[i] = pc - range_12h * 1.1 / 4
        r4_12h[i] = pc + range_12h * 1.1 / 2
        s4_12h[i] = pc - range_12h * 1.1 / 2
    
    # For first bar, use same values
    pivot_12h[0] = pivot_12h[1] if len(pivot_12h) > 1 else close_12h[0]
    r3_12h[0] = r3_12h[1] if len(r3_12h) > 1 else close_12h[0]
    s3_12h[0] = s3_12h[1] if len(s3_12h) > 1 else close_12h[0]
    r4_12h[0] = r4_12h[1] if len(r4_12h) > 1 else close_12h[0]
    s4_12h[0] = s4_12h[1] if len(s4_12h) > 1 else close_12h[0]
    
    # Pivot direction: 1 = bullish (price > pivot), -1 = bearish (price < pivot)
    pivot_dir_12h = np.where(close_12h > pivot_12h, 1, -1)
    
    # Align pivot direction and levels to 6h timeframe
    pivot_dir_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_dir_12h)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    
    # === 6h Indicators: Donchian Channel (20) ===
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 20)  # sufficient for Donchian and volume calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(pivot_dir_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~36h on 6h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if vol_ratio[i] > 1.5:  # Volume confirmation: require 1.5x average volume
            # Get regime from 12h pivot
            regime = pivot_dir_12h_aligned[i]
            
            # Long: Price breaks above Donchian high AND price > R4 (breakout confirmation) AND bullish regime
            if (price > donchian_high[i] and 
                price > r4_12h_aligned[i] and 
                regime > 0):
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Price breaks below Donchian low AND price < S4 (breakout confirmation) AND bearish regime
            elif (price < donchian_low[i] and 
                  price < s4_12h_aligned[i] and 
                  regime < 0):
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