#!/usr/bin/env python3
"""
Experiment #118: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation + ATR stoploss

HYPOTHESIS: Donchian channel breakouts on daily timeframe capture strong momentum moves, 
filtered by weekly HMA trend to avoid counter-trend trades, with volume confirmation ensuring 
institutional participation. This structure works in both bull and bear markets by trading 
with the higher timeframe trend. Targets 30-100 total trades over 4 years (7-25/year) to 
minimize fee drag while capturing high-probability breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian20_1w_hma_vol_v1"
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
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
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
    # Calculate Donchian(20) channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate ATR(14) for stoploss and position sizing
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Calculate volume ratio (current vs 20-period average)
    vol_ratio = np.full(n, 1.0)
    if n >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(20, 21)  # Ensure enough data for Donchian and HTF calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price crosses below HMA (trend change)
                if close[i] < hma_21_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price crosses above HMA (trend change)
                if close[i] > hma_21_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        # Long: Price breaks above Donchian HIGH with volume and price > HMA (uptrend)
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            close[i] > hma_21_aligned[i]
        )
        
        # Short: Price breaks below Donchian LOW with volume and price < HMA (downtrend)
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            close[i] < hma_21_aligned[i]
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