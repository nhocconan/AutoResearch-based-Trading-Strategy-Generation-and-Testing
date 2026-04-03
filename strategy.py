#!/usr/bin/env python3
"""
Experiment #2155: 6h Donchian(20) breakout + 1w Camarilla pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts capture swing momentum with weekly Camarilla pivot levels as 
institutional support/resistance. Volume confirmation filters false breakouts. Works in bull 
markets (trend continuation) and bear markets (mean reversion at extreme pivot levels R4/S4).
Primary: 6h, HTF: 1w for Camarilla pivot calculation, 1d for volume context.
Target: 75-150 total trades over 4 years (19-37/year) - optimized for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2155_6h_donchian20_1w_camarilla_vol_v1"
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
    
    # Calculate weekly Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = Pivot + (Range * 1.1/2)
    # R3 = Pivot + (Range * 1.1/4)
    # R2 = Pivot + (Range * 1.1/6)
    # R1 = Pivot + (Range * 1.1/12)
    # S1 = Pivot - (Range * 1.1/12)
    # S2 = Pivot - (Range * 1.1/6)
    # S3 = Pivot - (Range * 1.1/4)
    # S4 = Pivot - (Range * 1.1/2)
    
    pivot_1w = np.full_like(close_1w, np.nan)
    range_1w = np.full_like(close_1w, np.nan)
    r4_1w = np.full_like(close_1w, np.nan)
    r3_1w = np.full_like(close_1w, np.nan)
    s3_1w = np.full_like(close_1w, np.nan)
    s4_1w = np.full_like(close_1w, np.nan)
    
    for i in range(len(close_1w)):
        if i >= 1:  # Need previous week's data
            pivot_1w[i] = (high_1w[i-1] + low_1w[i-1] + close_1w[i-1]) / 3.0
            range_1w[i] = high_1w[i-1] - low_1w[i-1]
            r4_1w[i] = pivot_1w[i] + (range_1w[i] * 1.1 / 2)
            r3_1w[i] = pivot_1w[i] + (range_1w[i] * 1.1 / 4)
            s3_1w[i] = pivot_1w[i] - (range_1w[i] * 1.1 / 4)
            s4_1w[i] = pivot_1w[i] - (range_1w[i] * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # === HTF: 1d data for volume context (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # 1d volume moving average for context
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
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
            np.isnan(r4_1w_aligned[i]) or np.isnan(r3_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
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
                # Exit if price touches S3 Camarilla level (mean reversion)
                elif price <= s3_1w_aligned[i]:
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
                # Exit if price touches R3 Camarilla level (mean reversion)
                elif price >= r3_1w_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x 6h average) AND above 1d average
        volume_spike_6h = vol_ratio[i] > 1.5
        volume_context = volume[i] > vol_ma_1d_aligned[i] * 0.8  # Above 80% of 1d average
        
        if volume_spike_6h and volume_context:
            # Long entry: price breaks above upper Donchian AND above S4 (bullish bias)
            if price > donchian_upper[i] and price > s4_1w_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND below R4 (bearish bias)
            elif price < donchian_lower[i] and price < r4_1w_aligned[i]:
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