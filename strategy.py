#!/usr/bin/env python3
"""
Experiment #2167: 6h Donchian(20) breakout + 1d Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture momentum while 1d Camarilla levels (R3/S3 for fade, R4/S4 for breakout) provide institutional reference points. Volume confirmation ensures conviction. Works in bull markets via breakout continuation and bear markets via fading extremes at Camarilla levels.
- Primary: 6h Donchian(20) breakout with volume > 1.5x 20-bar average
- HTF: 1d Camarilla pivot levels (calculated from prior 1d bar) for entry/exit bias
- Exit: ATR(14) trailing stop (2*ATR) or opposite Camarilla level (R3/S3 for fade, R4/S4 for breakout)
- Target: 75-150 total trades over 4 years (19-37/year) - optimized for 6h timeframe
- Designed to work in both bull (breakout continuation) and bear (fade at extremes) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2167_6h_donchian20_1d_camarilla_vol_v1"
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
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Based on prior day's OHLC: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    camarilla_r4 = np.full_like(close_1d, np.nan)
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    camarilla_s4 = np.full_like(close_1d, np.nan)
    camarilla_pivot = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):  # Start from 1 to use prior day's data
        h = high_1d[i-1]
        l = low_1d[i-1]
        c = close_1d[i-1]
        range_hl = h - l
        
        camarilla_pivot[i] = (h + l + c) / 3
        camarilla_r4[i] = c + (range_hl * 1.1 / 2)
        camarilla_r3[i] = c + (range_hl * 1.1 / 4)
        camarilla_s3[i] = c - (range_hl * 1.1 / 4)
        camarilla_s4[i] = c - (range_hl * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 for completed bars only)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    
    # === 6h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                if price < highest_since_entry - 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches Camarilla S3 (fade level) or S4 (breakout failure)
                elif price <= camarilla_s3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                if price > lowest_since_entry + 2.0 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches Camarilla R3 (fade level) or R4 (breakout failure)
                elif price >= camarilla_r3_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine market regime relative to Camarilla levels
            # In extreme zones (beyond R4/S4), look for fade
            # In middle zones (between R3/S3), look for breakout continuation
            
            # Long entry conditions:
            # 1. Breakout continuation: price above Donchian upper AND between Camarilla R3 and R4
            # 2. Fade play: price below Camarilla S3 AND showing rejection (low close near low)
            if (price > donchian_upper[i] and 
                camarilla_r3_aligned[i] < price < camarilla_r4_aligned[i]):
                # Breakout continuation in middle zone
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif (price < camarilla_s3_aligned[i] and 
                  close[i] < (high[i] + low[i]) / 2):  # Weak close (near low)
                # Fade at extreme low
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            
            # Short entry conditions:
            # 1. Breakout continuation: price below Donchian lower AND between Camarilla S3 and R3
            # 2. Fade play: price above Camarilla R4 AND showing rejection (high close near high)
            elif (price < donchian_lower[i] and 
                  camarilla_s3_aligned[i] < price < camarilla_r3_aligned[i]):
                # Breakout continuation in middle zone
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            elif (price > camarilla_r4_aligned[i] and 
                  close[i] > (high[i] + low[i]) / 2):  # Strong close (near high)
                # Fade at extreme high
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals