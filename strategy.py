#!/usr/bin/env python3
"""
Experiment #331: 6h Camarilla Pivot + 1d Volume Spike + Regime Filter

HYPOTHESIS: Camarilla pivot levels from 1d timeframe provide high-probability reversal/continuation zones.
At 6h timeframe: 
- Long when price breaks above R4 with 1d volume spike and chop regime < 61.8 (trending)
- Short when price breaks below S4 with 1d volume spike and chop regime < 61.8 (trending)
- Exit on opposite Camarilla level (R3/S3) or ATR stoploss
Using 1d for pivot/volume/chop regime and 6f for execution minimizes false breakouts.
Target: 75-150 total trades over 4 years (19-37/year) to balance opportunity and fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_camarilla_vol_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 1d data for Camarilla pivots, volume, and chop regime (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 20:
        # Calculate Camarilla pivot levels for 1d
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        pivot = (high_1d + low_1d + close_1d) / 3
        range_1d = high_1d - low_1d
        
        # Camarilla levels
        r3 = pivot + range_1d * 1.1 / 2
        r4 = pivot + range_1d * 1.1
        s3 = pivot - range_1d * 1.1 / 2
        s4 = pivot - range_1d * 1.1
        
        # Align to 6h timeframe
        r3_6h = align_htf_to_ltf(prices, df_1d, r3)
        r4_6h = align_htf_to_ltf(prices, df_1d, r4)
        s3_6h = align_htf_to_ltf(prices, df_1d, s3)
        s4_6h = align_htf_to_ltf(prices, df_1d, s4)
        
        # 1d volume confirmation
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
        
        # 1d chop regime (Ehler's Chopiness Index)
        chop_1d = np.zeros(len(close_1d))
        atr_period = 14
        if len(close_1d) >= atr_period:
            tr_1d = np.zeros(len(close_1d))
            tr_1d[0] = high_1d[0] - low_1d[0]
            for j in range(1, len(close_1d)):
                tr_1d[j] = max(high_1d[j] - low_1d[j], 
                              abs(high_1d[j] - close_1d[j-1]), 
                              abs(low_1d[j] - close_1d[j-1]))
            atr_1d = pd.Series(tr_1d).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
            
            # Chop = 100 * log10(sum(atr1)/atr_period) / log10(atr_period)
            for i in range(atr_period, len(close_1d)):
                sum_atr = np.sum(atr_1d[i-atr_period:i])
                chop_1d[i] = 100 * np.log10(sum_atr / atr_period) / np.log10(atr_period)
            chop_1d[:atr_period] = 50.0  # neutral default
        
        chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    else:
        # Fallback if insufficient 1d data
        r3_6h = np.full(n, np.nan)
        r4_6h = np.full(n, np.nan)
        s3_6h = np.full(n, np.nan)
        s4_6h = np.full(n, np.nan)
        vol_ratio_1d_aligned = np.full(n, 1.0)
        chop_1d_aligned = np.full(n, 50.0)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (Chop < 61.8) ---
        is_trending = chop_1d_aligned[i] < 61.8
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Exit Logic ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at R3 (Camarilla)
                if close[i] >= r3_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at S3 (Camarilla)
                if close[i] <= s3_6h[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above R4 with volume confirmation in trending market
        long_condition = (
            close[i] > r4_6h[i] and 
            volume_spike and 
            is_trending
        )
        
        # Short: Break below S4 with volume confirmation in trending market
        short_condition = (
            close[i] < s4_6h[i] and 
            volume_spike and 
            is_trending
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals