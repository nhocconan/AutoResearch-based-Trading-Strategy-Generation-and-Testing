#!/usr/bin/env python3
"""
Experiment #1671: 6h Camarilla Pivot Breakout + 1d Volume Spike + ATR Stoploss
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for fade, R4/S4 for breakout) combined with 1d volume confirmation capture institutional order flow. In ranging markets, fade R3/S3 with volume spike; in trending markets, breakout R4/S4 with volume continuation. Uses 6h primary timeframe to balance trade frequency and signal quality. Target: 75-150 total trades over 4 years (19-38/year) by requiring both pivot level touch and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1671_6h_camarilla_breakout_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (Range * 1.1/2)
    # R3 = C + (Range * 1.1/4)
    # S3 = C - (Range * 1.1/4)
    # S4 = C - (Range * 1.1/2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + (range_1d * 1.1 / 2.0)
    r3_1d = close_1d + (range_1d * 1.1 / 4.0)
    s3_1d = close_1d - (range_1d * 1.1 / 4.0)
    s4_1d = close_1d - (range_1d * 1.1 / 2.0)
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed bars only)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
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
    
    warmup = 20  # sufficient for volume MA and ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
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
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Fade at R3/S3 (mean reversion)
            # Long when price touches S3 and starts bouncing up
            if abs(price - s3_aligned[i]) < (atr[i] * 0.1) and close[i] > open_time[i]:  # price near S3 and closing higher
                # Additional confirmation: price should be below pivot for fade long
                # We'll use close price vs pivot from 1d (need to align pivot)
                pivot_1d = (high_1d + low_1d + close_1d) / 3.0
                pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
                if not np.isnan(pivot_aligned[i]) and price < pivot_aligned[i]:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                    continue
            
            # Short when price touches R3 and starts rejecting down
            if abs(price - r3_aligned[i]) < (atr[i] * 0.1) and close[i] < open_time[i]:  # price near R3 and closing lower
                pivot_1d = (high_1d + low_1d + close_1d) / 3.0
                pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
                if not np.isnan(pivot_aligned[i]) and price > pivot_aligned[i]:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                    continue
            
            # Breakout continuation at R4/S4 (trend following)
            # Long breakout when price closes above R4 with volume
            if price > r4_aligned[i] and close[i] > open_time[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
                continue
            
            # Short breakdown when price closes below S4 with volume
            if price < s4_aligned[i] and close[i] < open_time[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
                continue
        
        signals[i] = 0.0
    
    return signals
)