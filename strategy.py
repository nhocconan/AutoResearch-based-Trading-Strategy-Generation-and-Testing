#!/usr/bin/env python3
"""
Experiment #056: 12h Donchian(20) Breakout + 1d Volume Spike + ATR Stoploss

HYPOTHESIS: 12h Donchian channel breakouts capture medium-term trends. 
Entries occur on breakouts above/below 20-period Donchian bands only when 
1d volume spikes (>2x 20-period average) indicate institutional participation. 
ATR-based stoploss (2.5x ATR) manages risk. Target: 75-150 trades over 4 years 
(19-37/year) with discrete sizing (0.25). Works in both bull (breakouts) and 
bear (breakdowns) markets via symmetric long/short logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian20_1d_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 12h Donchian Channel (20-period) ===
    # Calculate from 12h high/low (using current timeframe data)
    if n >= 20:
        # Donchian Upper: 20-period high
        donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Donchian Lower: 20-period low
        donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Align to current bar (no shift needed as we use completed 20-period window)
        donch_high_aligned = donch_high  # Already aligned to same index
        donch_low_aligned = donch_low
    else:
        donch_high_aligned = np.full(n, np.nan)
        donch_low_aligned = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 20  # Ensure enough data for Donchian
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: 1d volume spike > 2.0
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        if volume_spike:
            # Long breakout above Donchian Upper
            if close[i] > donch_high_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short breakdown below Donchian Lower
            elif close[i] < donch_low_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
        
        # No signal
        else:
            signals[i] = 0.0
    
    return signals