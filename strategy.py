#!/usr/bin/env python3
"""
Experiment #115: 6h Williams %R + 1d EMA + Volume Spike
HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h, 
while 1d EMA(50) provides the primary trend direction. Volume confirmation 
(>1.8x average) filters weak signals. This combination works in both bull 
and bear markets by taking mean-reversion entries in the direction of the 
higher timeframe trend. Discrete position sizing (0.25) minimizes fee churn. 
Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_115_6h_williamsr_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA(50) trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Indicators: Williams %R(14) ===
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.zeros(n)
    williams_r[:] = -100.0  # Initialize to -100 (oversold)
    for i in range(14, n):
        if highest_high_14[i] != lowest_low_14[i]:
            williams_r[i] = (highest_high_14[i] - close[i]) / (highest_high_14[i] - lowest_low_14[i]) * -100
        else:
            williams_r[i] = -50.0  # Neutral when range is zero
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Williams %R Conditions ---
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # --- 1d EMA Trend Direction ---
        uptrend = price > ema_50_1d_aligned[i]
        downtrend = price < ema_50_1d_aligned[i]
        
        # --- Exit Logic: Reverse signal or stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Exit conditions: reverse signal or stoploss (2*ATR approximation)
                if williams_r[i] > -20 or (i >= 1 and close[i] < entry_price - 0.02 * entry_price):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Exit conditions: reverse signal or stoploss (2*ATR approximation)
                if williams_r[i] < -80 or (i >= 1 and close[i] > entry_price + 0.02 * entry_price):
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if oversold and volume_spike and uptrend:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif overbought and volume_spike and downtrend:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals