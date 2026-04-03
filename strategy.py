#!/usr/bin/env python3
"""
Experiment #101: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation + ATR stoploss

HYPOTHESIS: Donchian channel breakouts capture strong momentum moves, confirmed by higher timeframe trend (1d HMA21) and volume spike (>2x average). This structure works in both bull and bear markets by trading breakouts in the direction of the higher timeframe trend. Targets 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to minimize fee drag while capturing high-probability trend continuation moves.
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
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, window):
            if len(arr) < window:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(arr, weights/weights.sum(), mode='valid')
        
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, 21)
        wma_2x_sub = 2 * wma_half - wma_full
        hma_21 = wma(wma_2x_sub, sqrt_len)
        
        # Pad to original length
        hma_padded = np.full(len(close_1d), np.nan)
        hma_padded[half_len:half_len + len(hma_21)] = hma_21
        hma_21_1d = hma_padded
        
        hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    else:
        hma_21_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian(20) channels
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        if i >= 20:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
        else:
            donchian_high[i] = high[i]
            donchian_low[i] = low[i]
    
    # Volume ratio (current vs 20-period average)
    vol_ratio = np.ones(n)
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
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
                # Update highest high for trailing stop (optional)
                highest_since_entry = max(highest_since_entry, high[i])
                # Take profit at Donchian low (breakdown below channel)
                if close[i] <= donchian_low[i]:
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
                # Update lowest low for trailing stop (optional)
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Take profit at Donchian high (breakout above channel)
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trend filter: price > 1d HMA21 for long, price < 1d HMA21 for short
        price_above_hma = close[i] > hma_21_1d_aligned[i]
        price_below_hma = close[i] < hma_21_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        # Long: Donchian breakout above upper channel with volume and trend alignment
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above Donchian high
            price_above_hma and              # Uptrend on 1d
            volume_spike                     # Volume confirmation
        )
        
        # Short: Donchian breakdown below lower channel with volume and trend alignment
        short_condition = (
            close[i] < donchian_low[i] and   # Breakdown below Donchian low
            price_below_hma and              # Downtrend on 1d
            volume_spike                     # Volume confirmation
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