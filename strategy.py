#!/usr/bin/env python3
"""
Experiment #323: 4h Donchian(20) Breakout + 12h HMA Trend + 1d Volume Spike

HYPOTHESIS: Donchian channel breakouts on 4h timeframe with HMA trend filter from 12h and 
volume confirmation from 1d creates a robust strategy that works in both bull and bear markets. 
The Donchian structure provides objective breakout levels, HMA trend filter ensures alignment 
with intermediate trend, and volume confirms institutional participation. Targets 75-200 
total trades over 4 years (19-50/year) to minimize fee drag while capturing high-probability 
breakouts with strong follow-through. Uses discrete position sizing (0.25) and ATR-based 
stoploss to control drawdown.
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
    
    # === HTF: 12h data for HMA trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h close
    if len(df_12h) >= 21:
        close_12h = df_12h['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, window):
            weights = np.arange(1, window + 1, dtype=np.float64)
            return np.convolve(arr, weights[::-1], mode='valid') / weights.sum()
        
        wma_half = np.full(len(close_12h), np.nan)
        wma_full = np.full(len(close_12h), np.nan)
        
        for i in range(len(close_12h)):
            if i >= half_len - 1:
                wma_half[i] = wma(close_12h[max(0, i-half_len+1):i+1], half_len)[-1]
            if i >= 21 - 1:
                wma_full[i] = wma(close_12h[max(0, i-21+1):i+1], 21)[-1]
        
        raw_hma = 2 * wma_half - wma_full
        hma_21 = np.full(len(close_12h), np.nan)
        for i in range(len(close_12h)):
            if i >= sqrt_len - 1 and not np.isnan(raw_hma[i]):
                hma_21[i] = wma(raw_half[max(0, i-sqrt_len+1):i+1], sqrt_len)[-1]
        
        hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
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
    
    # === 4h Indicators: Donchian Channel (20) ===
    # Calculate Donchian upper and lower bands (20-period high/low)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(n):
        if i >= 19:  # Need 20 periods for Donchian(20)
            start_idx = i - 19
            donchian_upper[i] = np.max(high[start_idx:i+1])
            donchian_lower[i] = np.min(low[start_idx:i+1])
        # Values remain NaN for i < 19 (handled in warmup)
    
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
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma_21_aligned[i]) or np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in alignment with 12h HMA21 trend ---
        price_above_hma = close[i] > hma_21_aligned[i]
        price_below_hma = close[i] < hma_21_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
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
                # Take profit at Donchian lower band (trailing stop for longs)
                if close[i] <= donchian_lower[i]:
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
                # Take profit at Donchian upper band (trailing stop for shorts)
                if close[i] >= donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above Donchian upper with volume and uptrend
        long_condition = (
            close[i] > donchian_upper[i] and 
            volume_spike and 
            price_above_hma
        )
        
        # Short: Break below Donchian lower with volume and downtrend
        short_condition = (
            close[i] < donchian_lower[i] and 
            volume_spike and 
            price_below_hma
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