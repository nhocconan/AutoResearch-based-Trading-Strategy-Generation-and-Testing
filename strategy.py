#!/usr/bin/env python3
"""
Experiment #419: 6h Williams %R + 12h Supertrend + Volume Confirmation

HYPOTHESIS: Williams %R(14) identifies overbought/oversold conditions on 6h timeframe, 
while 12h Supertrend provides the higher timeframe trend direction. Volume confirmation 
ensures institutional participation. This combination works in both bull and bear markets 
by taking mean-reversion entries at extremes in the direction of the 12h trend. Targets 
12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_williamsr_12h_supertrend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for Supertrend trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Supertrend on 12h
    if len(df_12h) >= 10:
        high_12h = df_12h['high'].values
        low_12h = df_12h['low'].values
        close_12h = df_12h['close'].values
        
        # ATR(10)
        tr12 = np.zeros(len(high_12h))
        tr12[0] = high_12h[0] - low_12h[0]
        for i in range(1, len(high_12h)):
            tr12[i] = max(high_12h[i] - low_12h[i], abs(high_12h[i] - close_12h[i-1]), abs(low_12h[i] - close_12h[i-1]))
        atr12 = pd.Series(tr12).ewm(span=10, min_periods=10, adjust=False).mean().values
        
        # Supertrend parameters
        atr_mult = 3.0
        hl2_12h = (high_12h + low_12h) / 2
        upper_band = hl2_12h + (atr_mult * atr12)
        lower_band = hl2_12h - (atr_mult * atr12)
        
        # Initialize Supertrend
        supertrend_12h = np.zeros(len(close_12h))
        direction_12h = np.ones(len(close_12h))  # 1 for uptrend, -1 for downtrend
        
        supertrend_12h[0] = upper_band[0]
        direction_12h[0] = 1
        
        for i in range(1, len(close_12h)):
            # Upper band logic
            if upper_band[i] < supertrend_12h[i-1] or close_12h[i-1] > supertrend_12h[i-1]:
                upper_band[i] = upper_band[i]
            else:
                upper_band[i] = supertrend_12h[i-1]
            
            # Lower band logic
            if lower_band[i] > supertrend_12h[i-1] or close_12h[i-1] < supertrend_12h[i-1]:
                lower_band[i] = lower_band[i]
            else:
                lower_band[i] = supertrend_12h[i-1]
            
            # Trend logic
            if direction_12h[i-1] == -1 and close_12h[i] > upper_band[i]:
                direction_12h[i] = 1
                supertrend_12h[i] = lower_band[i]
            elif direction_12h[i-1] == 1 and close_12h[i] < lower_band[i]:
                direction_12h[i] = -1
                supertrend_12h[i] = upper_band[i]
            elif direction_12h[i-1] == 1:
                direction_12h[i] = 1
                supertrend_12h[i] = max(lower_band[i], supertrend_12h[i-1])
            else:
                direction_12h[i] = -1
                supertrend_12h[i] = min(upper_band[i], supertrend_12h[i-1])
        
        # Align Supertrend direction to 6h timeframe
        direction_12h_aligned = align_htf_to_ltf(prices, df_12h, direction_12h)
    else:
        direction_12h_aligned = np.ones(n)  # Default to uptrend if insufficient data
    
    # === HTF: 12h data for volume spike (Call ONCE before loop) ===
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
    # === 6h Indicators ===
    # Williams %R(14)
    williams_r = np.full(n, np.nan)
    if n >= 14:
        highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        # Handle division by zero
        williams_r[highest_high == lowest_low] = -50
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(williams_r[i]) or 
            np.isnan(direction_12h_aligned[i]) or
            np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 1.5
        
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
        # Williams %R levels: oversold < -80, overbought > -20
        williams_oversold = williams_r[i] < -80
        williams_overbought = williams_r[i] > -20
        
        # Long: Williams %R oversold in 12h uptrend with volume spike
        long_condition = williams_oversold and (direction_12h_aligned[i] > 0) and volume_spike
        
        # Short: Williams %R overbought in 12h downtrend with volume spike
        short_condition = williams_overbought and (direction_12h_aligned[i] < 0) and volume_spike
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals