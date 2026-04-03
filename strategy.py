#!/usr/bin/env python3
"""
Experiment #268: 12h Donchian(20) Breakout + 1w HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts capture medium-term trends while 1w HMA(21) filters for primary trend alignment. Volume spike (>2x MA20) confirms institutional participation. ATR-based stoploss (2.5x) manages risk. This combination produces tight entries (~25-40 trades/year) with strong edge in both bull and bear markets by avoiding false breakouts in chop and capturing sustained moves. Targets 100-160 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_volume_vspike_v1"
timeframe = "12h"
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
        half = 21 // 2
        sqrt_n = int(np.sqrt(21))
        wma_half = pd.Series(close_1w).ewm(span=half, adjust=False).mean().values
        wma_full = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21_1w = pd.Series(raw_hma).ewm(span=sqrt_n, adjust=False).mean().values
        hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume MA (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume MA(20) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    else:
        vol_ma_20_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss or trend reversal) ---
        if in_position:
            # Stoploss: 2.5 * ATR against position
            if position_side > 0:  # Long
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price breaks below Donchian low (trend weakening)
                if close[i] < donch_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price breaks above Donchian high (trend weakening)
                if close[i] > donch_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- Volume Confirmation ---
        # Current 12h volume vs 1d volume MA(20) aligned
        # Need to estimate current 12h volume from 1d volume - approximation
        # Since we don't have true 12h volume MA, use volume > 1.5x 1d vol MA as proxy
        vol_spike = volume[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high + above 1w HMA + volume spike
        if (close[i] > donch_high[i] and 
            close[i] > hma_21_1w_aligned[i] and 
            vol_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Price breaks below Donchian low + below 1w HMA + volume spike
        elif (close[i] < donch_low[i] and 
              close[i] < hma_21_1w_aligned[i] and 
              vol_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals