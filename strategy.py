#!/usr/bin/env python3
"""
Experiment #013: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation + ATR stoploss

HYPOTHESIS: Donchian channel breakouts on 4h timeframe capture medium-term momentum. 
Combined with 12h Hull Moving Average for trend alignment and volume confirmation (>1.5x average), 
this strategy filters false breakouts. ATR-based stoploss (2.0x) manages risk. 
Designed to work in both bull (breakouts continue) and bear (breakdowns continue) markets 
by trading in direction of 12h trend. Target: 19-50 trades/year on 4h (75-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h close
    if len(df_12h) >= 21:
        close_12h = df_12h['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, window):
            if len(arr) < window:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        wma_half = wma(close_12h, half_len)
        wma_full = wma(close_12h, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_21 = wma(wma_diff, sqrt_len)
        
        # Pad to original length
        hma_21_padded = np.full(len(close_12h), np.nan)
        hma_21_padded[half_len:half_len + len(hma_21)] = hma_21
        hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21_padded)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === LTF: 4h data for Donchian channels and volume ===
    # Donchian(20) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume ratio (current vs 20-period average) on 4h
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros_like(volume)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
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
        if (np.isnan(hma_21_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in alignment with 12h HMA ---
        price_above_hma = close[i] > hma_21_aligned[i]
        price_below_hma = close[i] < hma_21_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper band with volume confirmation and uptrend
        long_condition = (
            close[i] > highest_high[i] and 
            volume_spike and 
            price_above_hma
        )
        
        # Short: Price breaks below Donchian lower band with volume confirmation and downtrend
        short_condition = (
            close[i] < lowest_low[i] and 
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

</think>