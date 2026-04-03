#!/usr/bin/env python3
"""
Experiment #1911: 6h Volume Spike + 1d Camarilla Pivot Breakout
HYPOTHESIS: Institutional traders defend Camarilla R3/S3 and R4/S4 levels from 1d timeframe. 
On 6h chart, volume spikes (>2x 20-bar average) breaking R4/S4 with 1d trend alignment 
indicate institutional participation. Enter breakout trades in direction of 1d trend. 
Exit when price reaches opposite Camarilla level (R3/S3) or shows weakening momentum.
Works in bull/bear by following 1d institutional flow. Target: 100-180 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1911_6h_volume_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels and trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + Range * 1.1/2
    # R3 = C + Range * 1.1/4
    # S3 = C - Range * 1.1/4
    # S4 = C - Range * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # Align 1d levels to 6h timeframe (shifted by 1 for completed bars only)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # 1d trend filter: EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
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
    bars_since_entry = 0
    
    warmup = 50  # sufficient for EMA(50) and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price reaches R3 (take profit at first resistance)
                if price >= r3_1d_aligned[i]:
                    exit_signal = True
                # Exit if volume drops significantly (weakening momentum)
                elif bars_since_entry >= 3 and vol_ratio[i] < 0.5:
                    exit_signal = True
            else:  # Short position
                # Exit if price reaches S3 (take profit at first support)
                if price <= s3_1d_aligned[i]:
                    exit_signal = True
                # Exit if volume drops significantly (weakening momentum)
                elif bars_since_entry >= 3 and vol_ratio[i] < 0.5:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require significant volume spike (> 2x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long entry: price breaks above R4 AND 1d trend up
            if trend_bias > 0 and price > r4_1d_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below S4 AND 1d trend down
            elif trend_bias < 0 and price < s4_1d_aligned[i]:
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
</file>