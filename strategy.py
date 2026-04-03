#!/usr/bin/env python3
"""
Experiment #1611: 6h Camarilla Pivot + 1d Volume + ADX Regime Filter
HYPOTHESIS: 6h Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) combined with 1d volume confirmation and ADX regime filter (trending vs ranging) captures high-probability reversals and continuations. In trending markets (ADX>25), R4/S4 breakouts are traded; in ranging markets (ADX<20), R3/S3 reversals are traded. This adaptive approach works in both bull and bear markets by aligning with the prevailing regime. Volume spike (>1.5x 20-period average) confirms institutional participation. Target: 75-150 total trades over 4 years (19-37/year) with discrete position sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1611_6h_camarilla_pivot_vol_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous 1d bar
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
    
    # Align HTF levels to LTF (6h) with shift(1) for completed bars only
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === LTF: 6h indicators ===
    # ADX(14) for regime detection
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.concatenate([[np.max([high[0]-low[0], np.abs(high[0]-close[0]), np.abs(low[0]-close[0])])], 
                            np.maximum(tr1, np.maximum(tr2, tr3))])
        # Directional Movement
        dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                           np.maximum(high[1:] - high[:-1], 0), 0)
        dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                            np.maximum(low[:-1] - low[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        # Smoothed TR, DM+
        atr = pd.Series(tr).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=period, adjust=False, min_periods=period).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=period, adjust=False, min_periods=period).mean().values
        # Directional Indicators
        dim_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        dim_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        # DX and ADX
        dx = 100 * np.abs(dim_plus - dim_minus) / (dim_plus + dim_minus + 1e-10)
        adx = pd.Series(dx).ewm(span=period, adjust=False, min_periods=period).mean().values
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    
    # Volume confirmation: volume spike (>1.5x 20-period average)
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
    
    warmup = 20  # sufficient for ADX and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss (using 2*ATR) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            tr[0] = high[0] - low[0]
            atr_val = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_val
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_val
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Regime detection: ADX > 25 = trending, ADX < 20 = ranging
        is_trending = adx[i] > 25
        is_ranging = adx[i] < 20
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            if is_trending:
                # Trending market: breakout continuation at R4/S4
                if price > r4_1d_aligned[i]:  # Break above R4
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                elif price < s4_1d_aligned[i]:  # Break below S4
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
            elif is_ranging:
                # Ranging market: mean reversion at R3/S3
                if price < r3_1d_aligned[i] and price > s3_1d_aligned[i]:
                    # In range, look for reversals from extremes
                    if price <= s3_1d_aligned[i] * 1.001:  # Near S3, go long
                        in_position = True
                        position_side = 1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = SIZE
                    elif price >= r3_1d_aligned[i] * 0.999:  # Near R3, go short
                        in_position = True
                        position_side = -1
                        entry_price = close[i]
                        bars_since_entry = 0
                        signals[i] = -SIZE
        
        # Default: no signal
        if not in_position:
            signals[i] = 0.0
    
    return signals