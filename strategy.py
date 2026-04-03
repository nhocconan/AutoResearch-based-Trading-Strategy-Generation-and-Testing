#!/usr/bin/env python3
"""
Experiment #033: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Confirmation + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts on 4h timeframe capture significant momentum moves, 
filtered by 12h HMA trend direction and volume confirmation. This combination provides 
high-probability entries with clear structure, minimizing false breakouts. Targeting 
75-200 total trades over 4 years (19-50/year) to balance opportunity with fee drag.
Works in both bull and bear markets by taking breakouts in direction of higher timeframe trend.
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
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights/weights.sum(), mode='valid')
        
        wma_half = np.full_like(close_12h, np.nan)
        wma_full = np.full_like(close_12h, np.nan)
        
        if len(close_12h) >= half_len:
            wma_half[half_len-1:] = wma(close_12h, half_len)
        if len(close_12h) >= 21:
            wma_full[20:] = wma(close_12h, 21)
        
        raw_hma = 2 * wma_half - wma_full
        hma_12h = np.full_like(close_12h, np.nan)
        if len(raw_hma) >= sqrt_len:
            hma_12h[sqrt_len-1:] = wma(raw_hma[sqrt_len-1:], sqrt_len)
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel (20) - highest high and lowest low of past 20 periods
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation - current volume vs 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-20:i])
    volume_ratio = np.divide(volume, vol_ma_20, out=np.ones(n), where=vol_ma_20!=0)
    
    # ATR(14) for stoploss calculation
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = np.full(n, np.nan)
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-14:i])
    
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
            np.isnan(hma_12h_aligned[i]) or np.isnan(volume_ratio[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: 12h HMA direction ---
        # Need previous HMA value to determine slope
        if i > warmup:
            hma_now = hma_12h_aligned[i]
            hma_prev = hma_12h_aligned[i-1]
            hma_rising = hma_now > hma_prev
            hma_falling = hma_now < hma_prev
        else:
            hma_rising = False
            hma_falling = False
        
        # --- Exit Logic (Trailing stop based on ATR) ---
        if in_position:
            if position_side > 0:  # Long position
                # Update highest since entry
                highest_since_entry = max(highest_since_entry, high[i])
                # Trail stop: exit if price drops 2.5*ATR from high
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
            else:  # Short position
                # Update lowest since entry
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Trail stop: exit if price rises 2.5*ATR from low
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    highest_since_entry = 0.0
                    lowest_since_entry = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian HIGH with volume and 12h HMA rising
        long_condition = (
            close[i] > donchian_high[i] and  # Breakout above channel
            volume_ratio[i] > 1.5 and        # Volume confirmation
            hma_rising                       # 12h trend up
        )
        
        # Short: Price breaks below Donchian LOW with volume and 12h HMA falling
        short_condition = (
            close[i] < donchian_low[i] and   # Breakdown below channel
            volume_ratio[i] > 1.5 and        # Volume confirmation
            hma_falling                      # 12h trend down
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
</p>