#!/usr/bin/env python3
# 6h_1w_1d_donchian_breakout_volume_v1
# Hypothesis: 6h Donchian(20) breakout with volume confirmation and 1w/1d trend alignment.
# Long: price breaks above 20-period high with volume > 2.0x 20-period average AND 1d close > 1w VWAP (bullish regime across timeframes)
# Short: price breaks below 20-period low with volume > 2.0x 20-period average AND 1d close < 1w VWAP (bearish regime across timeframes)
# Exit: price returns to 6h VWAP or opposite Donchian level with volume confirmation
# Uses 6h primary timeframe with 1d HTF for VWAP and 1w HTF for regime filter.
# Target: 75-150 total trades over 4 years (19-37/year) to balance opportunity and fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_donchian_breakout_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 6h volume ratio (current vs 20-period average)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    vol_ratio = np.where(vol_sma > 0, volume / vol_sma, 0)
    
    # Calculate 6h VWAP for exit
    vwap_6h = np.full(n, np.nan)
    typical_price = (high + low + close) / 3.0
    cum_vol = 0.0
    cum_vol_price = 0.0
    for i in range(n):
        cum_vol += volume[i]
        cum_vol_price += typical_price[i] * volume[i]
        if cum_vol > 0:
            vwap_6h[i] = cum_vol_price / cum_vol
    
    # Get 1d data for VWAP regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d VWAP
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    vwap_1d = np.full(len(df_1d), np.nan)
    cum_vol_1d = 0.0
    cum_vol_price_1d = 0.0
    for i in range(len(df_1d)):
        typical_price_1d = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        cum_vol_1d += volume_1d[i]
        cum_vol_price_1d += typical_price_1d * volume_1d[i]
        if cum_vol_1d > 0:
            vwap_1d[i] = cum_vol_price_1d / cum_vol_1d
    
    # Get 1w data for VWAP regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w VWAP
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    vwap_1w = np.full(len(df_1w), np.nan)
    cum_vol_1w = 0.0
    cum_vol_price_1w = 0.0
    for i in range(len(df_1w)):
        typical_price_1w = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
        cum_vol_1w += volume_1w[i]
        cum_vol_price_1w += typical_price_1w * volume_1w[i]
        if cum_vol_1w > 0:
            vwap_1w[i] = cum_vol_price_1w / cum_vol_1w
    
    # Align 1d VWAP to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Align 1w VWAP to 6h timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(50, n):
        vol_r = vol_ratio[i]
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vwap6h = vwap_6h[i]
        vwap1d = vwap_1d_aligned[i]
        vwap1w = vwap_1w_aligned[i]
        
        if np.isnan(vol_r) or np.isnan(upper) or np.isnan(lower) or np.isnan(vwap6h) or np.isnan(vwap1d) or np.isnan(vwap1w):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to 6h VWAP or breaks below Donchian low with volume
            if price <= vwap6h or (price < lower and vol_r > 1.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to 6h VWAP or breaks above Donchian high with volume
            if price >= vwap6h or (price > upper and vol_r > 1.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: price breaks above Donchian high with volume AND 1d VWAP > 1w VWAP (bullish regime)
            if price > upper and vol_r > 2.0 and vwap1d > vwap1w:
                position = 1
                entry_price = price
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume AND 1d VWAP < 1w VWAP (bearish regime)
            elif price < lower and vol_r > 2.0 and vwap1d < vwap1w:
                position = -1
                entry_price = price
                signals[i] = -0.25
    
    return signals