#!/usr/bin/env python3
"""
Experiment #1635: 6h Donchian(20) Breakout + 1w Camarilla Pivot + Volume Spike
HYPOTHESIS: 6h Donchian breakouts aligned with weekly Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) and volume spikes (>2x average) capture institutional order flow. Weekly pivot structure provides key support/resistance levels that price respects, while Donchian breakouts signal momentum. Volume confirmation ensures breakouts have conviction. Target: 75-150 total trades over 4 years (19-37/year) by requiring confluence of three filters. Works in bull/bear via pivot-based directional bias.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1635_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    # Camarilla levels: R4 = close + range * 1.5, R3 = close + range * 1.25
    # S3 = close - range * 1.25, S4 = close - range * 1.5
    camarilla_r3 = close_1w + range_1w * 1.25
    camarilla_r4 = close_1w + range_1w * 1.5
    camarilla_s3 = close_1w - range_1w * 1.25
    camarilla_s4 = close_1w - range_1w * 1.5
    
    # Align pivot levels to 6h timeframe (with shift(1) for completed weekly bar)
    r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # === 6h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = 20  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(r3_aligned[i]) or np.isnan(r4_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require significant volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Check for breakout at R4/S4 (continuation) or fade at R3/S3
            # Long conditions: price breaks above R4 OR fades from R3 with bullish bias
            long_breakout = price > donch_high[i] and price > r4_aligned[i]
            long_fade = price < donch_low[i] and price > s3_aligned[i] and price > r3_aligned[i]
            
            # Short conditions: price breaks below S4 OR fades from S3 with bearish bias
            short_breakout = price < donch_low[i] and price < s4_aligned[i]
            short_fade = price > donch_high[i] and price < r3_aligned[i] and price < s3_aligned[i]
            
            if long_breakout or long_fade:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif short_breakout or short_fade:
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