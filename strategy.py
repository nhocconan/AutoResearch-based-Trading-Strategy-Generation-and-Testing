#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R1/S1 breakout with 1d volume spike and ATR trailing stop
# Camarilla levels provide precise intraday support/resistance that work in ranging and trending markets.
# 1d volume spike confirms institutional interest in the breakout direction.
# ATR-based trailing stop manages risk without look-ahead.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
# Works in both bull and bear: breakouts capture trends, volume filter avoids false signals.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h data (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # === 1d data (HTF for volume confirmation) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    close_1d = df_1d['close'].values
    
    # === 6h Camarilla Pivot Levels (based on prior 6h bar) ===
    # Calculate from previous completed 6h bar (to avoid look-ahead)
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close = np.roll(close_6h, 1)
    prev_high[0] = high_6h[0]
    prev_low[0] = low_6h[0]
    prev_close[0] = close_6h[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_hl * 1.1 / 12)
    s1 = pivot - (range_hl * 1.1 / 12)
    r2 = pivot + (range_hl * 1.1 / 6)
    s2 = pivot - (range_hl * 1.1 / 6)
    r3 = pivot + (range_hl * 1.1 / 4)
    s3 = pivot - (range_hl * 1.1 / 4)
    r4 = pivot + (range_hl * 1.1 / 2)
    s4 = pivot - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_6h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_6h, s1)
    r4_aligned = align_htf_to_ltf(prices, df_6h, r4)
    s4_aligned = align_htf_to_ltf(prices, df_6h, s4)
    
    # === 1d Volume Confirmation (20-period average) ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # === 6h ATR (15) for trailing stop ===
    atr_6h = np.maximum(high_6h - low_6h, np.maximum(np.abs(high_6h - np.roll(close_6h, 1)), np.abs(low_6h - np.roll(close_6h, 1))))
    atr_6h[0] = high_6h[0] - low_6h[0]  # Fix first value
    atr_ma = pd.Series(atr_6h).ewm(span=15, adjust=False, min_periods=15).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_ma)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 100
    
    # Track position and entry price for stoploss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0  # For long trailing stop
    lowest_since_entry = 0.0   # For short trailing stop
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r4_val = r4_aligned[i]
        s4_val = s4_aligned[i]
        vol_ma_val = vol_ma_aligned[i]
        atr_val = atr_aligned[i]
        
        # Volume confirmation: current 6h volume > 1.5x 1d average volume
        vol_confirm = volume[i] > vol_ma_val * 1.5
        
        # === TRAILING STOP LOGIC ===
        if position == 1:  # Long position
            # Update highest price since entry
            if price > highest_since_entry:
                highest_since_entry = price
            # Trail stop: exit if price drops 2.5*ATR from high
            if price < highest_since_entry - 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Update lowest price since entry
            if price < lowest_since_entry:
                lowest_since_entry = price
            # Trail stop: exit if price rises 2.5*ATR from low
            if price > lowest_since_entry + 2.5 * atr_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === EXIT LOGIC (Camarilla reversal) ===
        if position == 1:  # Long position
            # Exit when price breaks below S1
            if price < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                highest_since_entry = 0.0
                continue
        
        elif position == -1:  # Short position
            # Exit when price breaks above R1
            if price > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
                lowest_since_entry = 0.0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume confirmation
            if vol_confirm:
                # Long when price breaks above R1 AND closes above R1 (breakout confirmation)
                if price > r1_val and close[i] > r1_val:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    highest_since_entry = price
                    continue
                # Short when price breaks below S1 AND closes below S1 (breakdown confirmation)
                elif price < s1_val and close[i] < s1_val:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    lowest_since_entry = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R1S1_Breakout_VolumeSpike_ATRTrail"
timeframe = "6h"
leverage = 1.0