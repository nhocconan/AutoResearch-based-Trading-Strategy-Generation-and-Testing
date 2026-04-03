#!/usr/bin/env python3
"""
Experiment #038: 1d Donchian(20) Breakout + 1w Volume Confirmation + ATR Stoploss

HYPOTHESIS: Daily Donchian channel breakouts (20-period) capture strong momentum moves.
Combined with weekly volume confirmation (>1.5x average) to filter false breakouts,
and ATR-based trailing stoploss (2.5x ATR) to manage risk. The weekly timeframe
provides structural context while minimizing overtrading. Targets 7-25 trades/year
on 1d timeframe (30-100 total over 4 years) to reduce fee drag and improve
generalization to bear markets like 2025+.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_1d_vol_1w_stoploss_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for volume confirmation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume ratio (current vs 20-period average) on 1w
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.zeros(len(vol_1w))
        vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w[:20] = 1.0  # Neutral for warmup
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # === Daily Donchian Channel (20-period) ===
    # Calculate highest high and lowest low over past 20 daily bars
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0  # For trailing stop
    lowest_since_entry = 0.0   # For trailing stop
    
    warmup = 50  # Ensure enough data for Donchian and HTF calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 1.5x weekly average) ---
        volume_spike = vol_ratio_1w_aligned[i] > 1.5
        
        # --- Exit Logic (ATR-based trailing stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss using available data up to i
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                # Update highest high since entry
                highest_since_entry = max(highest_since_entry, high[i])
                # Trailing stop: exit if price drops 2.5*ATR from highest since entry
                stop_level = highest_since_entry - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Update lowest low since entry
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Trailing stop: exit if price rises 2.5*ATR from lowest since entry
                stop_level = lowest_since_entry + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper band with volume confirmation
        long_condition = (
            close[i] > highest_high[i] and  # Breakout above 20-period high
            volume_spike                    # Volume confirmation
        )
        
        # Short: Price breaks below Donchian lower band with volume confirmation
        short_condition = (
            close[i] < lowest_low[i] and   # Breakdown below 20-period low
            volume_spike                   # Volume confirmation
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]  # Initialize tracking
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]  # Initialize tracking
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals