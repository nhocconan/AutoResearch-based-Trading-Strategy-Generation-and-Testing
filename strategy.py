#!/usr/bin/env python3
"""
Experiment #358: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation

HYPOTHESIS: Daily Donchian channel breakouts with weekly HMA trend filter and volume spike confirmation capture strong momentum moves in both bull and bear markets. The 1d timeframe reduces trade frequency to minimize fee drag, while weekly HMA ensures alignment with higher timeframe direction. Volume confirmation filters false breakouts. Targets 30-100 total trades over 4 years (7-25/year) to avoid overtrading.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "donchian_1d_vol_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1w).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21 = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian(20) channels
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 20:
            donchian_upper[i] = np.max(high[i-20:i])
            donchian_lower[i] = np.min(low[i-20:i])
        else:
            donchian_upper[i] = np.nan
            donchian_lower[i] = np.nan
    
    # Volume ratio (current vs 20-period average)
    vol_ratio = np.full(n, np.nan)
    if n >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio[20:] = volume[20:] / vol_ma[20:]
        vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(hma_21_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: HMA direction (rising for long, falling for short) ---
        if i >= warmup + 1:
            hma_rising = hma_21_aligned[i] > hma_21_aligned[i-1]
            hma_falling = hma_21_aligned[i] < hma_21_aligned[i-1]
        else:
            hma_rising = False
            hma_falling = False
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Exit Logic (Trailing stop based on ATR) ---
        if in_position:
            if position_side > 0:  # Long position
                # Update highest high since entry
                highest_since_entry = max(highest_since_entry, high[i])
                # Trail stop: exit if price drops 2.5*ATR from highest
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Also exit if price re-enters Donchian channel (failed breakout)
                if close[i] <= donchian_upper[i] and close[i] >= donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Update lowest low since entry
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Trail stop: exit if price rises 2.5*ATR from lowest
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Also exit if price re-enters Donchian channel
                if close[i] >= donchian_lower[i] and close[i] <= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above Donchian upper with volume + HMA rising
        long_condition = (
            close[i] > donchian_upper[i] and 
            volume_spike and 
            hma_rising
        )
        
        # Short: Break below Donchian lower with volume + HMA falling
        short_condition = (
            close[i] < donchian_lower[i] and 
            volume_spike and 
            hma_falling
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