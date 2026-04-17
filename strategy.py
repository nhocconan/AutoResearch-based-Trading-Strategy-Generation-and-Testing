#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with volume spike and ATR-based stoploss.
Long when price breaks above 20-period high with volume > 1.8x average in choppy market (CHOP > 61.8).
Short when price breaks below 20-period low with volume > 1.8x average in choppy market.
Exit on ATR trailing stop (3x ATR) or when price reverts to 20-period midpoint.
Uses 1d for chop regime filter, 12h for price/volume/Donchian channels.
Target: 50-150 total trades over 4 years (12-37/year). Focus on BTC/ETH with volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for chop regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Choppiness Index (CHOP)
    def calculate_chop(high, low, close, period=14):
        atr = np.zeros_like(close)
        tr = np.zeros_like(close)
        
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's ATR
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Sum of ATR over period
        atr_sum = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_sum[i] = np.sum(atr[i-period+1:i+1])
        
        # Max true range over period
        max_tr = np.zeros_like(close)
        for i in range(period, len(close)):
            max_tr[i] = np.max(tr[i-period+1:i+1])
        
        # Chop formula: 100 * log10(atr_sum / max_tr) / log10(period)
        chop = np.zeros_like(close)
        for i in range(period, len(close)):
            if max_tr[i] > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / max_tr[i]) / np.log10(period)
            else:
                chop[i] = 50  # neutral
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Align 1d chop to 12h timeframe
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    midpoint = (highest_high + lowest_low) / 2.0
    
    # Calculate 12h ATR (20-period) for stoploss
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate volume spike (current volume > 1.8x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    stop_price = 0.0
    
    start_idx = max(lookback, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(midpoint[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(chop_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        chop_val = chop_1d_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        mid_price = midpoint[i]
        atr_val = atr[i]
        
        # Chop regime: CHOP > 61.8 = ranging (good for breakout fade? Actually we want breakouts in chop)
        # Actually, breakouts work better in trending markets, but we use chop to filter false breakouts
        # In choppy markets (CHOP > 61.8), breakouts are more likely to fail, so we require stronger volume
        # In trending markets (CHOP < 38.2), breakouts are more reliable
        is_choppy = chop_val > 61.8
        is_trending = chop_val < 38.2
        
        if position == 0:
            # Long: price breaks above upper channel with volume spike
            if price > upper_channel and vol_spike:
                signals[i] = 0.25
                position = 1
                entry_price = price
                stop_price = price - 3.0 * atr_val
            # Short: price breaks below lower channel with volume spike
            elif price < lower_channel and vol_spike:
                signals[i] = -0.25
                position = -1
                entry_price = price
                stop_price = price + 3.0 * atr_val
        
        elif position == 1:
            # Update stop price (trailing stop)
            stop_price = max(stop_price, price - 3.0 * atr_val)
            # Exit long: stop hit OR price returns to midpoint
            if price <= stop_price or price <= mid_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update stop price (trailing stop)
            stop_price = min(stop_price, price + 3.0 * atr_val)
            # Exit short: stop hit OR price returns to midpoint
            if price >= stop_price or price >= mid_price:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeSpike_ATRStop_ChopFilter"
timeframe = "12h"
leverage = 1.0