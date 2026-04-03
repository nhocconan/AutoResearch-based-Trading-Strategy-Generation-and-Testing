#!/usr/bin/env python3
"""
Experiment #136: 12h Donchian(20) breakout + 1d volume spike + chop regime filter

HYPOTHESIS: Donchian(20) breakouts on 12h capture medium-term trends. Volume confirmation ensures
institutional participation. Choppiness index regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend)
adapts strategy: in trending regimes, trade breakouts; in ranging regimes, fade at Donchian bounds.
Uses 12h primary timeframe for lower trade frequency (target: 50-150 total trades over 4 years) and
1d HTF for regime/volume filters. Works in bull/bear via regime adaptation.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_chop_1d_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for regime and volume filters ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Choppiness Index (14-period) on 1d
    def calculate_chop(high, low, close, period=14):
        tr = np.zeros_like(high)
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr_sum = np.zeros_like(tr)
        for i in range(period, len(tr)):
            atr_sum[i] = np.sum(tr[i-period+1:i+1])
        
        hh = np.zeros_like(high)
        ll = np.zeros_like(low)
        for i in range(period-1, len(high)):
            hh[i] = np.max(high[i-period+1:i+1])
            ll[i] = np.min(low[i-period+1:i+1])
        
        chop = np.zeros_like(high)
        for i in range(period-1, len(high)):
            if hh[i] - ll[i] != 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
            else:
                chop[i] = 50.0
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 1d volume spike (volume > 1.5 * 20-period MA)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.5 * vol_ma_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(np.float64))
    
    # === 12h Indicators ===
    # Donchian channels (20-period)
    def donchian_channels(high, low, period=20):
        upper = np.zeros_like(high)
        lower = np.zeros_like(low)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    dc_upper, dc_lower = donchian_channels(high, low, 20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_1d_aligned[i]
        is_trending = chop_val < 38.2
        is_ranging = chop_val > 61.8
        vol_confirmed = vol_spike_aligned[i] > 0.5
        
        # --- Position Management (Exit Logic) ---
        if in_position:
            # Exit logic based on regime
            if is_trending:
                # In trend: trail with Donchian opposite band
                if position_side > 0:  # Long
                    if close[i] < dc_lower[i]:
                        in_position = False
                        position_side = 0
                else:  # Short
                    if close[i] > dc_upper[i]:
                        in_position = False
                        position_side = 0
            else:  # ranging
                # In range: fade at Donchian bounds (mean reversion)
                if position_side > 0:  # Long
                    if close[i] >= dc_upper[i]:
                        in_position = False
                        position_side = 0
                else:  # Short
                    if close[i] <= dc_lower[i]:
                        in_position = False
                        position_side = 0
            
            if not in_position:
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if is_trending and vol_confirmed:
            # Trend regime with volume: breakout entries
            if close[i] > dc_upper[i]:
                in_position = True
                position_side = 1
                signals[i] = SIZE
            elif close[i] < dc_lower[i]:
                in_position = True
                position_side = -1
                signals[i] = -SIZE
        elif is_ranging and vol_confirmed:
            # Range regime with volume: fade at Donchian bounds
            if close[i] >= dc_upper[i]:
                # Price at upper bound -> short (expect reversion to mean)
                in_position = True
                position_side = -1
                signals[i] = -SIZE
            elif close[i] <= dc_lower[i]:
                # Price at lower bound -> long (expect reversion to mean)
                in_position = True
                position_side = 1
                signals[i] = SIZE
    
    return signals

</think>