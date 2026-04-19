#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Donchian20_Breakout_VolumeATR_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels (once before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly high and low for Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels
    high_20_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian channels to daily timeframe
    high_20_1w_aligned = align_htf_to_ltf(prices, df_1w, high_20_1w)
    low_20_1w_aligned = align_htf_to_ltf(prices, df_1w, low_20_1w)
    
    # Weekly ATR for volatility filter (14-period)
    tr1 = np.maximum(high_1w[1:] - low_1w[1:], np.absolute(high_1w[1:] - close_1w[:-1]))
    tr1 = np.maximum(tr1, np.absolute(low_1w[1:] - close_1w[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])
    atr_14_1w = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_14_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average (daily)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(high_20_1w_aligned[i]) or np.isnan(low_20_1w_aligned[i]) or 
            np.isnan(atr_14_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        upper = high_20_1w_aligned[i]
        lower = low_20_1w_aligned[i]
        atr = atr_14_1w_aligned[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long: break above upper Donchian with volume
            if price > upper and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume
            elif price < lower and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price below lower Donchian or ATR-based stop
            if price < lower or price < close[i-1] - 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price above upper Donchian or ATR-based stop
            if price > upper or price > close[i-1] + 1.5 * atr:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals