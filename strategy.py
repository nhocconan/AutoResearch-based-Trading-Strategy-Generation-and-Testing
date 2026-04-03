#!/usr/bin/env python3
"""
Experiment #277: 4h Camarilla Pivot + Volume Spike + Choppiness Regime Filter

HYPOTHESIS: 4h price touches of Camarilla pivot levels (L3/L4 for longs, H3/H4 for shorts)
filtered by volume spikes (>2.0x average) and choppiness regime (CHOP > 61.8 = ranging)
capture mean-reversion bounces in ranging markets while avoiding trending environments.
Uses 1d HTF for higher-timeframe pivot calculation to ensure structure. Targets 19-50
trades/year (75-200 total over 4 years) with discrete position sizing (0.25) to minimize
fee drag. Works in both bull (bounces in ranges) and bear (failed breaks reverse) markets.
Includes ATR-based stoploss for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_277_4h_camarilla_pivot_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for 1d data
    def calculate_camarilla(high_arr, low_arr, close_arr):
        """Calculate Camarilla pivot levels: H4, H3, L3, L4"""
        if len(high_arr) < 1:
            return (np.full_like(high_arr, np.nan), np.full_like(high_arr, np.nan),
                    np.full_like(high_arr, np.nan), np.full_like(high_arr, np.nan))
        
        # Typical price for pivot
        typical_price = (high_arr + low_arr + close_arr) / 3.0
        # Pivot point
        pivot = typical_price
        # Range
        range_val = high_arr - low_arr
        
        # Camarilla levels
        h4 = pivot + (range_val * 1.1 / 2)
        h3 = pivot + (range_val * 1.1 / 4)
        l3 = pivot - (range_val * 1.1 / 4)
        l4 = pivot - (range_val * 1.1 / 2)
        
        return h4, h3, l3, l4
    
    h4_1d, h3_1d, l3_1d, l4_1d = calculate_camarilla(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Align HTF levels to LTF (4h)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === 4h Indicators: Choppiness Index (CHOP) for regime filter ===
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        """Calculate Choppiness Index: higher = ranging, lower = trending"""
        if len(high_arr) < period:
            return np.full_like(high_arr, 50.0)  # Neutral default
        
        atr_sum = np.zeros(n)
        for i in range(period, n):
            atr_sum[i] = np.sum(np.maximum(high_arr[i-period+1:i+1] - low_arr[i-period+1:i+1],
                                          np.maximum(np.abs(high_arr[i-period+1:i+1] - close_arr[i-period:i]),
                                                   np.abs(low_arr[i-period+1:i+1] - close_arr[i-period:i]))))
        
        # Avoid division by zero
        max_high = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        min_low = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        range_max_min = max_high - min_low
        
        chop = np.full(n, 50.0)
        mask = (range_max_min > 0) & (atr_sum > 0)
        chop[mask] = 100 * np.log10(atr_sum[mask] / range_max_min[mask]) / np.log10(period)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    chop_regime = chop > 61.8  # > 61.8 = ranging (mean revert), < 38.2 = trending
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 50  # Ensure enough data for HTF pivots, ATR, volume, CHOP
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(h4_1d_aligned[i]) or np.isnan(h3_1d_aligned[i]) or 
            np.isnan(l3_1d_aligned[i]) or np.isnan(l4_1d_aligned[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Regime Filter: Only trade in ranging markets (CHOP > 61.8) ---
        in_ranging_regime = chop_regime[i]
        
        # --- Camarilla Pivot Touch Conditions (with small tolerance) ---
        tolerance = 0.001  # 0.1% tolerance for level touch
        
        # Long: Touch L3 or L4 level from above (bounce up)
        touch_l3 = abs(close[i] - l3_1d_aligned[i]) <= (l3_1d_aligned[i] * tolerance)
        touch_l4 = abs(close[i] - l4_1d_aligned[i]) <= (l4_1d_aligned[i] * tolerance)
        long_condition = (touch_l3 or touch_l4) and close[i] > open[i] and volume_spike and in_ranging_regime
        
        # Short: Touch H3 or H4 level from below (bounce down)
        touch_h3 = abs(close[i] - h3_1d_aligned[i]) <= (h3_1d_aligned[i] * tolerance)
        touch_h4 = abs(close[i] - h4_1d_aligned[i]) <= (h4_1d_aligned[i] * tolerance)
        short_condition = (touch_h3 or touch_h4) and close[i] < open[i] and volume_spike and in_ranging_regime
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite pivot level touch (take profit)
                touch_h3 = abs(close[i] - h3_1d_aligned[i]) <= (h3_1d_aligned[i] * tolerance)
                touch_h4 = abs(close[i] - h4_1d_aligned[i]) <= (h4_1d_aligned[i] * tolerance)
                if touch_h3 or touch_h4:
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
                # Exit on opposite pivot level touch (take profit)
                touch_l3 = abs(close[i] - l3_1d_aligned[i]) <= (l3_1d_aligned[i] * tolerance)
                touch_l4 = abs(close[i] - l4_1d_aligned[i]) <= (l4_1d_aligned[i] * tolerance)
                if touch_l3 or touch_l4:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals