#!/usr/bin/env python3
"""
Experiment #168: 12h Donchian(20) Breakout + 1w Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe with volume confirmation (>2x 20-period average volume on 1w)
captures strong momentum moves while minimizing trade frequency. Uses ATR-based stoploss (2.5*ATR) to manage risk.
Target: 75-150 total trades over 4 years (19-38/year) - within winning range for 12h.
Uses weekly HTF for volume regime filter to avoid false breakouts in low-volume environments.
Works in bull/bear markets via volatility expansion breakouts that work regardless of trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_breakout_1w_volume_atr_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for volume spike filter ===
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    avg_vol_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_spike_1w = volume_1w > (2.0 * avg_vol_1w)
    vol_spike_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_spike_1w)
    
    # === 12h Indicators ===
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = -1
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_spike_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Donchian Breakout + Volume Confirmation ---
        # Upper breakout: price breaks above Donchian high with volume spike
        upper_breakout = (close[i] > donchian_high[i]) and vol_spike_1w_aligned[i]
        # Lower breakout: price breaks below Donchian low with volume spike
        lower_breakout = (close[i] < donchian_low[i]) and vol_spike_1w_aligned[i]
        
        # --- Position Management ---
        if in_position:
            # Check stoploss: 2.5 * ATR against position
            if position_side > 0:  # Long
                if close[i] < entry_price - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if close[i] > entry_price + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Still in position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: upper breakout + volume confirmation
        if upper_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        # Short: lower breakout + volume confirmation
        elif lower_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
    
    return signals

</think>