#!/usr/bin/env python3
"""
Experiment #2435: 6h Donchian(20) breakout + 1w Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with weekly Camarilla pivot structure (R4/S4 breakouts, R3/S3 fades) 
and volume spikes capture institutional participation. Works in bull markets (R4/S4 breakouts with volume) 
and bear markets (R3/S3 fade reversals with volume). Uses discrete position sizing (0.25) to limit fee drag 
and ensure 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2435_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for weekly
    # Camarilla: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We'll use: R4 = close + 1.1*(high-low)*1.1/2, S4 = close - 1.1*(high-low)*1.1/2
    # R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    rng_1w = high_1w - low_1w
    camarilla_r4 = close_1w + (rng_1w * 1.1 * 1.1 / 2)
    camarilla_r3 = close_1w + (rng_1w * 1.1 * 1.1 / 4)
    camarilla_s3 = close_1w - (rng_1w * 1.1 * 1.1 / 4)
    camarilla_s4 = close_1w - (rng_1w * 1.1 * 1.1 / 2)
    
    # Align HTF Camarilla levels to 6h timeframe
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # === 6h Indicators: Donchian(20) channels, Volume MA(20) ===
    # Donchian channels (20-period high/low)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA for spike detection
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry (using Donchian width as ATR proxy)
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15  # approximate ATR from channel width
                if price < highest_since_entry - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below Donchian low (mean reversion)
                elif price < lowest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                donchian_width = highest_20[i] - lowest_20[i]
                atr_estimate = donchian_width * 0.15
                if price > lowest_since_entry + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above Donchian high (mean reversion)
                elif price > highest_20[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long entry conditions:
            # 1. Breakout above weekly R4 (strong bullish continuation)
            # 2. Fade from weekly S3 (bullish reversal from support)
            if (price > highest_20[i] and price > r4_1w_aligned[i]) or \
               (price > lowest_20[i] and price < s3_1w_aligned[i] and price > open[i]):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry conditions:
            # 1. Breakdown below weekly S4 (strong bearish continuation)
            # 2. Fade from weekly R3 (bearish reversal from resistance)
            elif (price < lowest_20[i] and price < s4_1w_aligned[i]) or \
                 (price < highest_20[i] and price > r3_1w_aligned[i] and price < open[i]):
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