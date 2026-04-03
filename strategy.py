#!/usr/bin/env python3
"""
Experiment #085: 12h Donchian(20) breakout + 1d HMA trend + volume confirmation

HYPOTHESIS: Donchian channel breakouts on 12h timeframe, filtered by 1d HMA(21) trend direction and 
12h volume spike (>2.0x average), captures strong momentum moves in both bull and bear markets. 
The Donchian structure provides objective breakout levels, HMA filter ensures alignment with higher 
timeframe trend to avoid counter-trend whipsaws, and volume confirmation filters weak breakouts. 
Targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to minimize fee drag while 
participating in significant market moves. ATR-based stoploss manages risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian20_1d_hma_vol_v1"
timeframe = "12h"
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
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
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
        raw_hma = 2 * wma_half - wma_full
        hma_21_1d = np.array([wma(raw_hma[i:i+sqrt_len], sqrt_len) 
                             if i+sqrt_len <= len(raw_hma) and not np.isnan(raw_hma[i:i+sqrt_len]).any() 
                             else np.nan 
                             for i in range(len(raw_hma))])
        hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    else:
        hma_21_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume average (Call ONCE before loop) ===
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    else:
        vol_ma_20_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Calculate Donchian(20) channels on 12h using previous 20 periods
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        # Get the last 20 completed 12h bars before current bar
        start_idx = max(0, i - 20)
        donchian_high[i] = np.max(high[start_idx:i])
        donchian_low[i] = np.min(low[start_idx:i])
    
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
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: HMA direction ---
        hma_rising = hma_21_1d_aligned[i] > hma_21_1d_aligned[i-1] if i > 0 else False
        hma_falling = hma_21_1d_aligned[i] < hma_21_1d_aligned[i-1] if i > 0 else False
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = volume[i] > 2.0 * vol_ma_20_aligned[i]
        
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
        # Long: Price breaks above Donchian High with HMA rising and volume spike
        long_condition = (
            close[i] > donchian_high[i] and 
            hma_rising and 
            volume_spike
        )
        
        # Short: Price breaks below Donchian Low with HMA falling and volume spike
        short_condition = (
            close[i] < donchian_low[i] and 
            hma_falling and 
            volume_spike
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