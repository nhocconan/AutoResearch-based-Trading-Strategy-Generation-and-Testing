#!/usr/bin/env python3
"""
Experiment #203: 4h Donchian(20) Breakout + 12h Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian(20) breakouts on 4h with volume confirmation (>2x 20-period average volume on 12h)
captures strong momentum moves. Uses ATR-based stoploss (2.5*ATR) to manage risk. 
Target: 100-180 total trades over 4 years (25-45/year) - within winning range for 4h.
Works in bull/bear markets via volatility expansion breakouts that work regardless of trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_breakout_12h_volume_atr_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike filter ===
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    avg_vol_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (2.0 * avg_vol_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # === 4h Indicators ===
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
            np.isnan(vol_spike_12h_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Donchian Breakout + Volume Confirmation ---
        # Upper breakout: price breaks above Donchian high with volume spike
        upper_breakout = (close[i] > donchian_high[i]) and vol_spike_12h_aligned[i]
        # Lower breakout: price breaks below Donchian low with volume spike
        lower_breakout = (close[i] < donchian_low[i]) and vol_spike_12h_aligned[i]
        
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