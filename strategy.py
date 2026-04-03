#!/usr/bin/env python3
"""
Experiment #081: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Confirmation + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, filtered by 1d Hull Moving Average trend direction and 
1d volume spike confirmation, captures strong momentum moves while avoiding false breakouts in choppy markets. 
The strategy uses discrete position sizing (0.25) to minimize fee drag and includes ATR-based stoploss for risk management. 
Target: 75-200 trades over 4 years (19-50/year) on 4h timeframe to balance opportunity with cost efficiency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        
        wma_half = np.array([wma(close_1d[i:i+half_len], half_len) 
                            if i+half_len <= len(close_1d) else np.nan 
                            for i in range(len(close_1d))])
        wma_full = np.array([wma(close_1d[i:i+21], 21) 
                            if i+21 <= len(close_1d) else np.nan 
                            for i in range(len(close_1d))])
        hma_21_raw = 2 * wma_half - wma_full
        hma_21 = np.array([wma(hma_21_raw[i:i+sqrt_len], sqrt_len) 
                          if i+sqrt_len <= len(hma_21_raw) else np.nan 
                          for i in range(len(hma_21_raw))])
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
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
        if (np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Calculate Donchian(20) on 4h ---
        lookback = 20
        start_idx = max(0, i - lookback + 1)
        highest_high = np.max(high[start_idx:i+1])
        lowest_low = np.min(low[start_idx:i+1])
        
        # --- Regime Filter: Only trade in alignment with 1d HMA trend ---
        price_above_hma = close[i] > hma_21_aligned[i]
        price_below_hma = close[i] < hma_21_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.8
        
        # --- Exit Logic (ATR-based stoploss and Donchian opposite breakout) ---
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
                stop_level = highest_since_entry - 2.5 * atr_14
                # Exit conditions: stoploss OR Donchian lower band break (contrarian exit)
                if low[i] < stop_level or close[i] < lowest_low:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
            else:  # Short position
                # Update lowest since entry
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_level = lowest_since_entry + 2.5 * atr_14
                # Exit conditions: stoploss OR Donchian upper band break (contrarian exit)
                if high[i] > stop_level or close[i] > highest_high:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper band with volume and trend alignment
        long_condition = (
            close[i] > highest_high and 
            price_above_hma and 
            volume_spike
        )
        
        # Short: Price breaks below Donchian lower band with volume and trend alignment
        short_condition = (
            close[i] < lowest_low and 
            price_below_hma and 
            volume_spike
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