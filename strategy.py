#!/usr/bin/env python3
"""
4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation + ATR(14) stoploss
Hypothesis: Donchian breakouts capture momentum moves. 12h HMA21 filters trend direction to avoid counter-trend whipsaws. Volume confirmation ensures breakout validity. ATR-based trailing stop limits drawdown in bear markets. Designed for 4h timeframe targeting 75-200 trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA21 trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA21 for trend
    close_12h = pd.Series(df_12h['close'])
    hma_21_12h = calculate_hma(close_12h.values, 21)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Calculate 4h ATR(14) for stoploss and position sizing
    atr_14 = np.full(n, np.nan)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-13:i+1])
    
    # Calculate 20-period volume MA for volume spike confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Start index: need enough for Donchian, HMA, ATR, volume MA
    start_idx = max(20, 21, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_12h_aligned[i]) or np.isnan(atr_14[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        donchian_high_val = donchian_high[i]
        donchian_low_val = donchian_low[i]
        hma_21_val = hma_21_12h_aligned[i]
        atr_14_val = atr_14[i]
        vol_ma = vol_ma_20[i]
        
        # Trend filter: price relative to 12h HMA21
        uptrend = curr_close > hma_21_val
        downtrend = curr_close < hma_21_val
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirm = curr_volume > 1.5 * vol_ma
        
        if position == 0:
            # Look for breakout signals
            # Long: price breaks above Donchian high with volume confirmation in uptrend
            long_breakout = (curr_close > donchian_high_val) and volume_confirm and uptrend
            # Short: price breaks below Donchian low with volume confirmation in downtrend
            short_breakout = (curr_close < donchian_low_val) and volume_confirm and downtrend
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                highest_since_entry = curr_close
            elif short_breakout:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                lowest_since_entry = curr_close
            else:
                signals[i] = 0.0
        elif position == 1:
            # Update highest price since entry
            highest_since_entry = max(highest_since_entry, curr_high)
            # ATR trailing stop: exit if price drops 2.5 * ATR from highest since entry
            if curr_close < highest_since_entry - 2.5 * atr_14_val:
                signals[i] = 0.0
                position = 0
            # Exit long: price closes below Donchian low OR HMA21 trend turns down
            elif curr_close < donchian_low_val or curr_close < hma_21_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest price since entry
            lowest_since_entry = min(lowest_since_entry, curr_low)
            # ATR trailing stop: exit if price rises 2.5 * ATR from lowest since entry
            if curr_close > lowest_since_entry + 2.5 * atr_14_val:
                signals[i] = 0.0
                position = 0
            # Exit short: price closes above Donchian high OR HMA21 trend turns up
            elif curr_close > donchian_high_val or curr_close > hma_21_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

def calculate_hma(values, period):
    """Calculate Hull Moving Average"""
    if len(values) < period:
        return np.full_like(values, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = np.full_like(values, np.nan)
    for i in range(half_period - 1, len(values)):
        wma_half[i] = np.average(values[i - half_period + 1:i + 1], 
                                weights=np.arange(1, half_period + 1))
    
    # WMA of full period
    wma_full = np.full_like(values, np.nan)
    for i in range(period - 1, len(values)):
        wma_full[i] = np.average(values[i - period + 1:i + 1], 
                                weights=np.arange(1, period + 1))
    
    # Raw HMA: 2 * WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of raw HMA with sqrt_period
    hma = np.full_like(values, np.nan)
    for i in range(sqrt_period - 1, len(values)):
        hma[i] = np.average(raw_hma[i - sqrt_period + 1:i + 1], 
                           weights=np.arange(1, sqrt_period + 1))
    
    return hma

name = "4h_Donchian20_Breakout_12hHMA21_Trend_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0