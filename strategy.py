#!/usr/bin/env python3
"""
Experiment #146: 4h Camarilla Pivot + Volume Spike + Choppiness Regime Filter

HYPOTHESIS: Camarilla pivot levels from 1d timeframe act as strong support/resistance zones. 
Price touching these levels with volume confirmation (>1.5x average) and in favorable 
choppiness regime (CHOP > 50 indicates ranging market conducive to mean reversion from pivots) 
provides high-probability mean-reversion trades. The strategy uses discrete position sizing 
(0.25) to limit fee impact and includes ATR-based stoploss for risk control. 
Designed to work in both bull and bear markets by fading extremes at institutional pivot levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_camarilla_pivot_vol_chop_v1"
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
    
    # Calculate Camarilla pivot levels for 1d
    if len(df_1d) >= 2:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Camarilla levels: based on previous day's range
        # R4 = Close + (High - Low) * 1.1/2
        # R3 = Close + (High - Low) * 1.1/4
        # R2 = Close + (High - Low) * 1.1/6
        # R1 = Close + (High - Low) * 1.1/12
        # S1 = Close - (High - Low) * 1.1/12
        # S2 = Close - (High - Low) * 1.1/6
        # S3 = Close - (High - Low) * 1.1/4
        # S4 = Close - (High - Low) * 1.1/2
        
        range_1d = high_1d - low_1d
        camarilla_r4 = close_1d + range_1d * 1.1 / 2
        camarilla_r3 = close_1d + range_1d * 1.1 / 4
        camarilla_r2 = close_1d + range_1d * 1.1 / 6
        camarilla_r1 = close_1d + range_1d * 1.1 / 12
        camarilla_s1 = close_1d - range_1d * 1.1 / 12
        camarilla_s2 = close_1d - range_1d * 1.1 / 6
        camarilla_s3 = close_1d - range_1d * 1.1 / 4
        camarilla_s4 = close_1d - range_1d * 1.1 / 2
        
        # Align all levels to LTF
        r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
        r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
        r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
        r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
        s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
        s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
        s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    else:
        r4_aligned = r3_aligned = r2_aligned = r1_aligned = np.full(n, np.nan)
        s1_aligned = s2_aligned = s3_aligned = s4_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Choppiness Index (CHOP) for regime filter ===
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        """Calculate Choppiness Index: higher values = more ranging/choppy market"""
        atr_sum = np.zeros(n)
        max_high = np.zeros(n)
        min_low = np.zeros(n)
        
        # Calculate True Range
        tr = np.zeros(n)
        tr[0] = high_arr[0] - low_arr[0]
        for i in range(1, n):
            tr[i] = max(high_arr[i] - low_arr[i], 
                       abs(high_arr[i] - close_arr[i-1]), 
                       abs(low_arr[i] - close_arr[i-1]))
        
        # Calculate ATR sum over period
        for i in range(period, n):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
            max_high[i] = np.max(high_arr[i-period+1:i+1])
            min_low[i] = np.min(low_arr[i-period+1:i+1])
        
        # CHOP = 100 * log10(atr_sum / (max_high - min_low)) / log10(period)
        chop = np.full(n, 50.0)  # Default neutral value
        for i in range(period, n):
            if max_high[i] > min_low[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (max_high[i] - min_low[i])) / np.log10(period)
            else:
                chop[i] = 50.0
        return chop
    
    chop = calculate_chop(high, low, close, period=14)
    
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
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(chop[i]) or np.isnan(atr_14[i]) or 
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in choppy/ranging markets (CHOP > 50) ---
        chop_regime = chop[i] > 50.0
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Pivot Touch Conditions (with small tolerance) ---
        tolerance = 0.001 * close[i]  # 0.1% tolerance for level touch
        
        # Touch resistance levels (R1, R2, R3, R4) for short entries
        touch_resistance = (abs(high[i] - r1_aligned[i]) <= tolerance or 
                           abs(high[i] - r2_aligned[i]) <= tolerance or
                           abs(high[i] - r3_aligned[i]) <= tolerance or
                           abs(high[i] - r4_aligned[i]) <= tolerance)
        
        # Touch support levels (S1, S2, S3, S4) for long entries
        touch_support = (abs(low[i] - s1_aligned[i]) <= tolerance or 
                        abs(low[i] - s2_aligned[i]) <= tolerance or
                        abs(low[i] - s3_aligned[i]) <= tolerance or
                        abs(low[i] - s4_aligned[i]) <= tolerance)
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite pivot level
                if position_side > 0 and close[i] >= r1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite pivot level
                if position_side < 0 and close[i] <= s1_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Touch support + volume spike + choppy regime
        long_condition = touch_support and volume_spike and chop_regime
        
        # Short: Touch resistance + volume spike + choppy regime
        short_condition = touch_resistance and volume_spike and chop_regime
        
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