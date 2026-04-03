#!/usr/bin/env python3
"""
Experiment #261: 4h Camarilla Pivot + Volume Spike + Chop Regime Filter

HYPOTHESIS: Camarilla pivot levels from 1d timeframe act as strong support/resistance zones.
When price breaks above/below pivot levels with volume confirmation (>2.0x average) and
market is in trending regime (Choppiness Index < 38.2 on 4h), we enter in breakout direction.
In ranging regimes (Choppiness Index > 61.8), we avoid trading. Uses ATR-based stoploss (2.5x)
and minimum 4-bar holding period to reduce churn. Target: 75-200 trades over 4 years.
Works in both bull/bear markets by trading breakouts with institutional volume at key levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_261_4h_camarilla_pivot_vol_chop_v1"
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
    
    # Calculate 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H4 = close + range * 1.1/2, H3 = close + range * 1.1/4, etc.
    camarilla_h4 = close_1d + range_1d * 1.1 / 2.0
    camarilla_h3 = close_1d + range_1d * 1.1 / 4.0
    camarilla_l3 = close_1d - range_1d * 1.1 / 4.0
    camarilla_l4 = close_1d - range_1d * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # === 4h Indicators: Choppiness Index (14) for regime detection ===
    def calculate_choppiness(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high[0] - low[0]  # First period TR
        
        # Sum of TR over period
        tr_sum = pd.Series(tr).rolling(window=period, min_periods=period).sum().values
        
        # Highest high and lowest low over period
        hh = pd.Series(high).rolling(window=period, min_periods=period).max().values
        ll = pd.Series(low).rolling(window=period, min_periods=period).min().values
        
        # Choppiness Index formula
        chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(period)
        # Handle division by zero when hh == ll
        chop[hh == ll] = 100.0
        return chop
    
    chop = calculate_choppiness(high, low, close, 14)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0  # Track bars in position for minimum holding period
    
    warmup = 100  # Warmup for stability
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Regime Filter: Choppiness Index ---
        # CHOP < 38.2 = trending regime (trade breakouts)
        # CHOP > 61.8 = ranging regime (avoid breakouts)
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Camarilla Breakout Conditions ---
        # Break above H3 or H4 (bullish)
        breakout_h3 = price > camarilla_h3_aligned[i-1]
        breakout_h4 = price > camarilla_h4_aligned[i-1]
        # Break below L3 or L4 (bearish)
        breakout_l3 = price < camarilla_l3_aligned[i-1]
        breakout_l4 = price < camarilla_l4_aligned[i-1]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Camarilla breakout (contrarian exit)
                if breakout_l3 and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit on opposite Camarilla breakout (contrarian exit)
                if breakout_h3 and volume_spike:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 4 bars to reduce churn
            if bars_since_entry < 4:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Only trade in trending regimes (CHOP < 38.2)
        if is_trending:
            # Long: Break above H3/H4 AND volume spike
            if (breakout_h3 or breakout_h4) and volume_spike:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Break below L3/L4 AND volume spike
            elif (breakout_l3 or breakout_l4) and volume_spike:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            # In ranging regime or transition, do not trade breakouts
            signals[i] = 0.0
    
    return signals