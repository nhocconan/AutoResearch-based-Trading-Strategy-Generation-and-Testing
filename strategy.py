#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Donchian20_VolumeTrend_V2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR for trend filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr2])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d EMA34 for trend direction
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(atr_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_1d_aligned[i]
        ema = ema34_1d_aligned[i]
        
        # Volume filter
        volume_ok = vol > 1.8 * vol_ma
        
        # Trend filter: price above/below EMA34
        price_above_ema = price > ema
        price_below_ema = price < ema
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and uptrend
            if price > high_roll[i] and volume_ok and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and downtrend
            elif price < low_roll[i] and volume_ok and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns below Donchian low or trend reverses
            if price < low_roll[i] or price < ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns above Donchian high or trend reverses
            if price > high_roll[i] or price > ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals