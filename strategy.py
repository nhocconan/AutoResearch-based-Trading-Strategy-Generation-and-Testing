#!/usr/bin/env python3
"""
Experiment #691: 6h Williams %R Extreme + 1d Supertrend + Volume Confirmation
HYPOTHESIS: Williams %R identifies overextended conditions on 6h, while 1d Supertrend provides the primary trend filter. 
In bear markets (2022-2024), extreme %R readings (< -90 or > -10) often precede mean-reversion bounces aligned with the 1d trend. 
Volume confirmation (>1.8x average) ensures institutional participation. Designed for 6h timeframe to achieve 75-200 total trades 
over 4 years (19-50/year). Works in both bull and bear markets: long when %R < -90 and price > Supertrend, short when %R > -10 
and price < Supertrend. Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_691_6h_williamsr_extreme_1d_supertrend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Supertrend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Supertrend for 1d timeframe
    # ATR(10)
    tr_1d = np.zeros(len(high_1d))
    for i in range(1, len(high_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], abs(high_1d[i] - close_1d[i-1]), abs(low_1d[i] - close_1d[i-1]))
    tr_1d[0] = high_1d[0] - low_1d[0]
    atr_1d = pd.Series(tr_1d).ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # Basic Upper/Lower Bands
    hl2_1d = (high_1d + low_1d) / 2.0
    upper_band_1d = hl2_1d + (3.0 * atr_1d)
    lower_band_1d = hl2_1d - (3.0 * atr_1d)
    
    # Supertrend calculation
    supertrend_1d = np.zeros(len(close_1d))
    direction_1d = np.ones(len(close_1d))  # 1 for uptrend, -1 for downtrend
    
    supertrend_1d[0] = upper_band_1d[0]
    direction_1d[0] = 1
    
    for i in range(1, len(close_1d)):
        if close_1d[i] > supertrend_1d[i-1]:
            direction_1d[i] = 1
        elif close_1d[i] < supertrend_1d[i-1]:
            direction_1d[i] = -1
        else:
            direction_1d[i] = direction_1d[i-1]
        
        if direction_1d[i] == 1:
            supertrend_1d[i] = max(lower_band_1d[i], supertrend_1d[i-1])
        else:
            supertrend_1d[i] = min(upper_band_1d[i], supertrend_1d[i-1])
    
    # Align Supertrend and direction to 6h timeframe
    supertrend_1d_aligned = align_htf_to_ltf(prices, df_1d, supertrend_1d)
    direction_1d_aligned = align_htf_to_ltf(prices, df_1d, direction_1d)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.zeros(n)
    for i in range(n):
        if highest_high[i] == lowest_low[i]:
            williams_r[i] = -50.0  # Avoid division by zero
        else:
            williams_r[i] = (highest_high[i] - close[i]) / (highest_high[i] - lowest_low[i]) * -100.0
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Williams %R and Supertrend calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(williams_r[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(supertrend_1d_aligned[i]) or np.isnan(direction_1d_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry (wider for 6h)
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~48h on 6h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Williams %R < -90 (extreme oversold) + price > Supertrend (1d uptrend)
            if williams_r[i] < -90.0 and direction_1d_aligned[i] > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Williams %R > -10 (extreme overbought) + price < Supertrend (1d downtrend)
            elif williams_r[i] > -10.0 and direction_1d_aligned[i] < 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals