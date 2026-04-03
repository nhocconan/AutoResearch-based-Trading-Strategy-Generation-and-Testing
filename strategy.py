#!/usr/bin/env python3
"""
Experiment #1927: 6h Donchian(20) Breakout + 1d Weekly Pivot Direction + Volume Spike + ATR Stoploss
HYPOTHESIS: 6h Donchian breakouts capture swing momentum. Filter by 1d weekly pivot (Camarilla R4/S4) direction to align with institutional levels, requiring volume spike (>2x avg) to confirm breakout strength. ATR trailing stop (2.5x) manages risk. Targets 75-200 total trades over 4 years (19-50/year) to balance opportunity with fee drag. Works in bull/bear via breakout/breakdown with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1927_6h_donchian20_1d_pivot_vol_sl_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points (Camarilla) from prior 1d bar
    # Using prior bar to avoid look-ahead: R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    # Actually Camarilla: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We'll use R4/S4 as breakout/breakdown levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    rang = high_1d - low_1d
    r4 = pivot + (rang * 1.1 / 2.0)
    s4 = pivot - (rang * 1.1 / 2.0)
    
    # Determine bias: if close > pivot, bullish bias (look for longs at R4 breaks)
    # if close < pivot, bearish bias (look for shorts at S4 breaks)
    bias_1d = np.where(close_1d > pivot, 1, -1)
    bias_1d_aligned = align_htf_to_ltf(prices, df_1d, bias_1d)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # === 6h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR calculation
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
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
    
    warmup = 50  # sufficient for Donchian(20), ATR(14)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i]) or
            np.isnan(bias_1d_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # --- Exit Logic (Trailing Stoploss) ---
        if in_position:
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume spike filter: require > 2.0x average volume
        volume_spike = vol_ratio > 2.0
        
        if volume_spike:
            # Long entry: price breaks above Donchian HIGH AND above 1d R4 + bullish bias
            if bias_1d_aligned[i] > 0 and price > donch_high[i] and price > r4_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian LOW AND below 1d S4 + bearish bias
            elif bias_1d_aligned[i] < 0 and price < donch_low[i] and price < s4_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals