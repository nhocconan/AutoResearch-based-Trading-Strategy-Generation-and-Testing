#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trailing stop.
Long when price breaks above 20-period Donchian high with volume > 1.8x average.
Short when price breaks below 20-period Donchian low with volume > 1.8x average.
Exit on ATR(14) trailing stop (3x ATR from extreme price) or opposite breakout.
Uses 1d for volume average calculation to reduce noise, 4h for price action.
Target: 100-180 total trades over 4 years (25-45/year) to stay within fee-efficient range.
Donchian channels provide clear structure, volume confirms conviction, ATR stop manages risk in volatile markets.
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
    
    # Get 1d data for volume average (less noisy than 4h volume MA)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on 4h
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 1d volume average (20-period) and align to 4h
    volume_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    # Calculate ATR(14) for trailing stop
    atr_period = 14
    tr = np.zeros(n)
    atr = np.full(n, np.nan)
    
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    # Wilder's ATR
    if n > atr_period:
        atr[atr_period] = np.mean(tr[1:atr_period+1])
        for i in range(atr_period+1, n):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    long_stop = 0.0
    short_stop = 0.0
    
    start_idx = max(lookback, atr_period) + 5
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(volume_ma_1d_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ma = volume_ma_1d_aligned[i]
        atr_val = atr[i]
        vol_spike = volume[i] > (vol_ma * 1.8)
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike
            if price > donchian_high[i] and vol_spike:
                signals[i] = 0.25
                position = 1
                long_stop = price - 3.0 * atr_val  # Initial stop
            # Short: price breaks below Donchian low with volume spike
            elif price < donchian_low[i] and vol_spike:
                signals[i] = -0.25
                position = -1
                short_stop = price + 3.0 * atr_val  # Initial stop
        
        elif position == 1:
            # Update trailing stop for long
            long_stop = max(long_stop, price - 3.0 * atr_val)
            
            # Exit long: price hits stop OR breaks below Donchian low
            if price <= long_stop or price < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update trailing stop for short
            short_stop = min(short_stop, price + 3.0 * atr_val)
            
            # Exit short: price hits stop OR breaks above Donchian high
            if price >= short_stop or price > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0