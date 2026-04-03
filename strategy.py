#!/usr/bin/env python3
"""
Experiment #370: 1d Donchian(20) breakout + HMA trend (1w) + volume confirmation + ATR stoploss

HYPOTHESIS: Daily Donchian(20) breakouts capture significant momentum moves. Using 1w HMA(21) as trend filter ensures we only trade in the direction of the higher timeframe trend, reducing whipsaws. Volume confirmation (>1.5x 20-day average) adds conviction. ATR-based stoploss (2.5x) manages risk. Targets 20-40 trades/year on 1d timeframe (80-160 total over 4 years) for minimal fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_hma_vol_v1"
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
    
    # Calculate HMA(21) on 1w
    if len(df_1w) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        close_1w = df_1w['close'].values
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        if len(wma_half) >= half_len and len(wma_full) >= 21:
            raw_hma = 2 * wma_half[-len(wma_full):] - wma_full
            hma_21 = wma(raw_hma, sqrt_len)
            # Pad to match original length
            hma_21_padded = np.full(len(close_1w), np.nan)
            hma_21_padded[half_len - 1 + len(wma_half) - len(raw_hma):half_len - 1 + len(wma_half) - len(raw_hma) + len(hma_21)] = hma_21
            hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21_padded)
        else:
            hma_21_aligned = np.full(n, np.nan)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
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
        if np.isnan(hma_21_aligned[i]):
            signals[i] = 0.0
            continue
        
        # --- Calculate Donchian(20) on daily data up to i ---
        lookback = min(20, i+1)
        if lookback < 20:
            signals[i] = 0.0
            continue
            
        highest_high = np.max(high[i-lookback+1:i+1])
        lowest_low = np.min(low[i-lookback+1:i+1])
        
        # --- Volume Confirmation: > 1.5x 20-day average ---
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-20:i])
            volume_spike = volume[i] > (vol_ma_20 * 1.5)
        else:
            volume_spike = False
        
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
                # Take profit at Donchian upper band (trailing)
                if close[i] >= highest_high:
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
                # Take profit at Donchian lower band (trailing)
                if close[i] <= lowest_low:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper band with volume and above 1w HMA
        long_condition = (
            close[i] > highest_high and 
            volume_spike and 
            close[i] > hma_21_aligned[i]
        )
        
        # Short: Price breaks below Donchian lower band with volume and below 1w HMA
        short_condition = (
            close[i] < lowest_low and 
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