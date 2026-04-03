#!/usr/bin/env python3
"""
Experiment #022: 12h Donchian(20) Breakout + 1d Volume Spike + 1w Choppiness Regime

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe capture medium-term trends. 
Volume confirmation from 1d timeframe ensures institutional participation. 
Choppiness regime filter from 1w timeframe avoids whipsaws in ranging markets (CHOP > 61.8) 
and only allows breakouts in trending conditions (CHOP < 38.2). 
Discrete position sizing (0.25) and ATR-based stoploss manage risk. 
Target: 50-150 trades over 4 years on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian20_1d_vol_1w_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike confirmation (Call ONCE before loop) ===
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
    
    # === HTF: 1w data for choppiness regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index(14) on 1w
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr_1w = np.zeros(len(close_1w))
        tr_1w[0] = high_1w[0] - low_1w[0]
        for i in range(1, len(close_1w)):
            tr_1w[i] = max(high_1w[i] - low_1w[i], 
                          abs(high_1w[i] - close_1w[i-1]), 
                          abs(low_1w[i] - close_1w[i-1]))
        
        # Sum of TR over 14 periods
        tr_sum_14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh_14 = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
        ll_14 = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index
        chop_1w = np.zeros(len(close_1w))
        for i in range(len(close_1w)):
            if tr_sum_14[i] > 0 and hh_14[i] > ll_14[i]:
                chop_1w[i] = 100 * np.log10(tr_sum_14[i] / (hh_14[i] - ll_14[i])) / np.log10(14)
            else:
                chop_1w[i] = 50.0  # Neutral when undefined
        
        chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    else:
        chop_1w_aligned = np.full(n, 50.0)  # Neutral default
    
    # === Primary: Donchian(20) breakout on 12h timeframe ===
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    if n >= lookback:
        for i in range(lookback - 1, n):
            window_high = high[i - lookback + 1:i + 1]
            window_low = low[i - lookback + 1:i + 1]
            highest_high[i] = np.max(window_high)
            lowest_low[i] = np.min(window_low)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, lookback, 20)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Donchian Breakout Conditions ---
        breakout_up = close[i] > highest_high[i-1]  # Break above previous period's high
        breakout_down = close[i] < lowest_low[i-1]   # Break below previous period's low
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Choppiness Regime Filter: Only allow breakouts in trending markets (CHOP < 38.2) ---
        trending_market = chop_1w_aligned[i] < 38.2
        
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
        long_condition = (
            breakout_up and 
            volume_spike and 
            trending_market
        )
        
        short_condition = (
            breakout_down and 
            volume_spike and 
            trending_market
        )
        
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