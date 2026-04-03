#!/usr/bin/env python3
"""
Experiment #919: 6h Donchian(20) + 12h Camarilla Pivot + Volume Spike
HYPOTHESIS: Donchian breakouts on 6h capture momentum, filtered by 12h Camarilla pivot levels 
(R3/S3 for mean reversion fade, R4/S4 for breakout continuation) and volume confirmation. 
Long when price breaks above Donchian upper AND at/below S3 (fade) OR at/above R4 (continuation) 
AND volume spike. Short when price breaks below Donchian lower AND at/above R3 (fade) OR at/below S4 (continuation) 
AND volume spike. Uses discrete position sizing (0.25) to manage risk. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_919_6h_donchian20_12h_camarilla_vol_v1"
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
    
    # Calculate Camarilla pivot levels for previous 12h bar
    # Camarilla: R4 = close + ((high - low) * 1.1/2), R3 = close + ((high - low) * 1.1/4)
    #          S3 = close - ((high - low) * 1.1/4), S4 = close - ((high - low) * 1.1/2)
    rng = high_12h - low_12h
    camarilla_r4 = close_12h + (rng * 1.1 / 2)
    camarilla_r3 = close_12h + (rng * 1.1 / 4)
    camarilla_s3 = close_12h - (rng * 1.1 / 4)
    camarilla_s4 = close_12h - (rng * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (using previous completed 12h bar)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # === 6h Indicators: Donchian Channel (20) ===
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channel(high, low, 20)
    
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
    
    warmup = max(20, 20)  # sufficient for Donchian, volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(r4_12h_aligned[i]) or
            np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(s4_12h_aligned[i]) or np.isnan(atr[i])):
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
            
            # Optional: time-based exit after 4 bars (~24h on 6h) to avoid overtrading
            if bars_since_entry > 4:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.7x average)
        volume_spike = vol_ratio[i] > 1.7
        
        if volume_spike:
            # Long conditions:
            # 1. Fade at S3: price breaks above Donchian upper AND price <= S3 (mean reversion)
            # 2. Continuation at R4: price breaks above Donchian upper AND price >= R4 (breakout)
            long_fade = price > upper_20[i] and price <= s3_12h_aligned[i]
            long_continuation = price > upper_20[i] and price >= r4_12h_aligned[i]
            
            # Short conditions:
            # 1. Fade at R3: price breaks below Donchian lower AND price >= R3 (mean reversion)
            # 2. Continuation at S4: price breaks below Donchian lower AND price <= S4 (breakdown)
            short_fade = price < lower_20[i] and price >= r3_12h_aligned[i]
            short_continuation = price < lower_20[i] and price <= s4_12h_aligned[i]
            
            if long_fade or long_continuation:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif short_fade or short_continuation:
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