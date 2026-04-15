#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d Donchian channels (20)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    
    # 1d ADX (14) for trend strength
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(
        high_1d[1:] - low_1d[1:],
        np.maximum(np.abs(high_1d[1:] - high_1d[:-1]), np.abs(low_1d[1:] - low_1d[:-1]))
    )
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(np.concatenate([[np.nan], plus_dm])).rolling(window=14, min_periods=14).mean().values / atr_14
    minus_di = 100 * pd.Series(np.concatenate([[np.nan], minus_dm])).rolling(window=14, min_periods=14).mean().values / atr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_14 = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: close breaks above 1d Donchian high + ADX > 25 + volume confirmation
        if (close[i] > donch_high_aligned[i] and 
            adx_aligned[i] > 25 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: close breaks below 1d Donchian low + ADX > 25 + volume confirmation
        elif (close[i] < donch_low_aligned[i] and 
              adx_aligned[i] > 25 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: close crosses back inside Donchian bands (mean reversion)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < donch_high_aligned[i]) or
               (signals[i-1] == -0.25 and close[i] > donch_low_aligned[i]))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_1D_Donchian_Breakout_ADX_Volume"
timeframe = "4h"
leverage = 1.0