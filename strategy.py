#!/usr/bin/env python3
"""
Experiment #386: 4h Donchian Breakout + 1d Volume Spike + 1d HMA Trend

HYPOTHESIS: Donchian(20) breakout on 4h timeframe, confirmed by 1d volume spike (>2x average) 
and filtered by 1d HMA(21) trend, creates a robust strategy for both bull and bear markets. 
The Donchian structure captures breakouts with clear risk definition, volume confirms 
institutional participation, and HMA trend filter ensures alignment with higher timeframe 
direction to avoid counter-trend trades. Targets 19-50 trades/year on 4h timeframe 
(75-200 total over 4 years) to minimize fee drag while capturing high-probability breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_hma_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike and HMA trend (Call ONCE before loop) ===
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
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        # WMA function
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = np.array([wma(close_1d[i:i+half_len], half_len) 
                            if i+half_len <= len(close_1d) else np.nan 
                            for i in range(len(close_1d))])
        wma_full = np.array([wma(close_1d[i:i+21], 21) 
                            if i+21 <= len(close_1d) else np.nan 
                            for i in range(len(close_1d))])
        wma_sqrt = np.array([wma(close_1d[i:i+sqrt_len], sqrt_len) 
                            if i+sqrt_len <= len(close_1d) else np.nan 
                            for i in range(len(close_1d))])
        
        hma_21 = 2 * wma_half - wma_full
        hma_21 = np.array([wma(wma_sqrt[i:i+sqrt_len], sqrt_len) 
                          if i+sqrt_len <= len(wma_sqrt) and not np.isnan(wma_sqrt[i]) 
                          else np.nan 
                          for i in range(len(wma_sqrt))])
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Calculate Donchian(20) channels
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    
    for i in range(n):
        if i >= 20:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
        else:
            donchian_high[i] = high[i]
            donchian_low[i] = low[i]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in alignment with 1d HMA trend ---
        price_above_hma = close[i] > hma_21_aligned[i]
        price_below_hma = close[i] < hma_21_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss and Donchian exit) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                # Update highest since entry
                highest_since_entry = max(highest_since_entry, high[i])
                
                # Stoploss: 2.5 * ATR below entry
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                
                # Exit: Price closes below Donchian low (failed breakout)
                if close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                    
            else:  # Short position
                # Update lowest since entry
                lowest_since_entry = min(lowest_since_entry, low[i])
                
                # Stoploss: 2.5 * ATR above entry
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                
                # Exit: Price closes above Donchian high (failed breakdown)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high with volume and trend alignment
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above upper band
            volume_spike and                 # Volume confirmation
            price_above_hma                  # Trend filter: above HMA
        )
        
        # Short: Price breaks below Donchian low with volume and trend alignment
        short_condition = (
            close[i] < donchian_low[i] and   # Breakdown below lower band
            volume_spike and                 # Volume confirmation
            price_below_hma                  # Trend filter: below HMA
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals