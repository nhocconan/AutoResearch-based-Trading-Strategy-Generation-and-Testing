#!/usr/bin/env python3
"""
12h_WilliamsAlligator_Regime_Breakout_v1
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) defines trend regime on 1w HTF. Enter breakouts of 12h Donchian(10) in direction of Alligator alignment, confirmed by 1d volume spike (>2.0). Uses ATR(14) trailing stop (3.0x) for risk control. Designed for low frequency (15-25 trades/year) to minimize fee drag and work in both bull/bear regimes via trend-following logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === Load HTF data ONCE before loop ===
    df_1w = get_htf_data(prices, '1w')  # Weekly for Alligator trend regime
    df_1d = get_htf_data(prices, '1d')  # Daily for volume confirmation
    
    if len(df_1w) < 40 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1w Williams Alligator (Smoothed Medians) ===
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    close_1w = df_1w['close'].values
    jaw_raw = pd.Series(close_1w).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(close_1w).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(close_1w).rolling(window=5, min_periods=5).mean().values
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Align to LTF (12h)
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # === 1d volume average (20-period) for spike detection ===
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d[np.isnan(vol_ma_1d)] = 1.0
    vol_ratio_1d = volume_1d / vol_ma_1d
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    # === 12h Donchian channels (10-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    highest_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # === ATR for dynamic trailing stop (14-period on 12h) ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(atr_14[i]) or
            np.isnan(highest_10[i]) or np.isnan(lowest_10[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_spike = vol_ratio_1d_aligned[i]
        upper_donchian = highest_10[i]
        lower_donchian = lowest_10[i]
        atr_val = atr_14[i]
        
        # Define Alligator alignment: Lips > Teeth > Jaw = bullish, Lips < Teeth < Jaw = bearish
        alligator_bullish = lips_val > teeth_val and teeth_val > jaw_val
        alligator_bearish = lips_val < teeth_val and teeth_val < jaw_val
        
        if position == 0:
            # Long: price breaks above Donchian + bullish Alligator + volume spike > 2.0
            if price_close > upper_donchian and alligator_bullish and vol_spike > 2.0:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
                highest_since_entry = price_close
            # Short: price breaks below Donchian + bearish Alligator + volume spike > 2.0
            elif price_close < lower_donchian and alligator_bearish and vol_spike > 2.0:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
                lowest_since_entry = price_close
        
        elif position != 0:
            # Update highest/lowest since entry for trailing stop
            if position == 1:
                highest_since_entry = max(highest_since_entry, price_high)
                # Trailing stop: 3.0 * ATR below highest since entry
                if price_close < highest_since_entry - 3.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                lowest_since_entry = min(lowest_since_entry, price_low)
                # Trailing stop: 3.0 * ATR above lowest since entry
                if price_close > lowest_since_entry + 3.0 * atr_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Regime_Breakout_v1"
timeframe = "12h"
leverage = 1.0