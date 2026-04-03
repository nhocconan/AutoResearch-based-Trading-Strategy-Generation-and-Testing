#!/usr/bin/env python3
"""
Experiment #409: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Confirmation

HYPOTHESIS: Donchian channel breakouts on 4h timeframe, confirmed by 1d HMA trend direction 
and volume spike (>2x average), capture strong momentum moves in both bull and bear markets. 
The Donchian structure provides clear breakout levels, HMA filter ensures alignment with 
higher timeframe trend to avoid counter-trend whipsaws, and volume confirmation filters 
false breakouts. Targets 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
Uses discrete position sizing (0.25) and ATR-based stoploss (2.5x ATR) for risk management.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_vol_v1"
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
        hma_21 = 2 * wma_half - wma_full
        hma_21 = np.array([wma(hma_21[i:i+sqrt_len], sqrt_len) 
                          if i+sqrt_len <= len(hma_21) else np.nan 
                          for i in range(len(hma_21))])
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel (20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume ratio (current vs 20-period average)
    vol_ratio = np.full(n, 1.0)
    if n >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR(14) for stoploss
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: HMA direction ---
        hma_rising = hma_21_aligned[i] > hma_21_aligned[i-1]
        hma_falling = hma_21_aligned[i] < hma_21_aligned[i-1]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian low break (trailing stop)
                if close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit on Donchian high break (trailing stop)
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout above upper band with volume and HMA rising
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            hma_rising
        )
        
        # Short: Donchian breakdown below lower band with volume and HMA falling
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            hma_falling
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