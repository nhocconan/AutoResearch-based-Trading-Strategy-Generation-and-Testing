#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_VolumeSpike_ATRFilter_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Donchian channels (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # ATR (14-period) for volatility filter
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.absolute(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(atr14[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema200 = ema200_1d_aligned[i]
        atr = atr14[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        # Trend filter: price above EMA200 for long, below for short
        uptrend = price > ema200
        downtrend = price < ema200
        
        if position == 0:
            # Long: break above Donchian high with volume and uptrend
            if price > high_20[i] and volume_confirmed and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and downtrend
            elif price < low_20[i] and volume_confirmed and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below Donchian low or ATR-based stop
            if price < low_20[i] or price < close[i-1] - 2.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above Donchian high or ATR-based stop
            if price > high_20[i] or price > close[i-1] + 2.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals