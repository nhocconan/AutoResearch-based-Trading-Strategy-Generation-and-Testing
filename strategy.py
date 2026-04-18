#!/usr/bin/env python3
"""
4h Donchian Breakout + 1d ATR Filter + Volume Spike
Strategy: Enter long when price breaks above 4h Donchian(20) high with volume spike
          and 1d ATR > 1d ATR(30) mean (volatility regime). Enter short when price
          breaks below 4h Donchian(20) low with volume spike and same volatility filter.
          Exit when price returns to 4h Donchian(20) midpoint.
          Uses 1d ATR to avoid low-volatility chop and focus on breakout strength.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ATR(14) for volatility filter
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # True Range components
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.roll(daily_close, 1))
    tr3 = np.abs(daily_low - np.roll(daily_close, 1))
    tr1[0] = 0  # first value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) with Wilder's smoothing (equivalent to EMA with alpha=1/14)
    atr_14 = np.zeros_like(tr)
    atr_14[13] = np.mean(tr[0:14])  # seed with simple average
    for i in range(14, len(tr)):
        atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # ATR(30) mean for volatility regime filter
    atr_30_mean = np.zeros_like(atr_14)
    for i in range(29, len(atr_14)):
        atr_30_mean[i] = np.mean(atr_14[i-29:i+1])
    
    # Volatility filter: ATR(14) > ATR(30) mean (high volatility regime)
    vol_filter = atr_14 > atr_30_mean
    
    # Align daily volatility filter to 4h timeframe
    vol_filter_aligned = align_htf_to_ltf(prices, df_1d, vol_filter)
    
    # 4h Donchian channel (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2.0
    
    # Volume spike detection (1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 20)  # need enough data for ATR and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        mid = donchian_mid[i]
        vol_ok = vol_filter_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume spike and high vol regime
            if price > upper and volume_spike[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike and high vol regime
            elif price < lower and volume_spike[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price returns to Donchian midpoint
            if price <= mid:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price returns to Donchian midpoint
            if price >= mid:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_ATRFilter_VolumeSpike"
timeframe = "4h"
leverage = 1.0