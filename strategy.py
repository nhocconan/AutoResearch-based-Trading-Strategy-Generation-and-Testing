#!/usr/bin/env python3
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
    
    # Get 1d data ONCE before loop
    daily = get_htf_data(prices, '1d')
    close_d = daily['close'].values
    high_d = daily['high'].values
    low_d = daily['low'].values
    
    # 1d ATR(14) for volatility filter (minimum 14 periods)
    tr1 = np.maximum(high_d[1:] - low_d[1:], np.abs(high_d[1:] - close_d[:-1]))
    tr2 = np.maximum(np.abs(low_d[1:] - close_d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])
    atr_14d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14d_aligned = align_htf_to_ltf(prices, daily, atr_14d)
    
    # 1d Donchian channels (20-period)
    donch_high = np.full(len(close_d), np.nan)
    donch_low = np.full(len(close_d), np.nan)
    for i in range(20, len(close_d)):
        donch_high[i] = np.max(high_d[i-20:i])
        donch_low[i] = np.min(low_d[i-20:i])
    donch_high_aligned = align_htf_to_ltf(prices, daily, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, daily, donch_low)
    
    # 6h ATR(10) for position sizing (volatility scaling)
    tr1_6h = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2_6h = np.maximum(np.abs(low[1:] - close[:-1]), tr1_6h)
    tr_6h = np.concatenate([[np.nan], tr2_6h])
    atr_10_6h = pd.Series(tr_6h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Volatility filter: 1d ATR > 0.5 * 6h ATR (avoid low volatility periods)
    vol_filter = atr_14d_aligned > (0.5 * atr_10_6h)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(atr_14d_aligned[i]) or np.isnan(vol_filter[i])):
            continue
        
        # Long: Price breaks above 1d Donchian high + volatility filter
        if (close[i] > donch_high_aligned[i] and 
            vol_filter[i]):
            signals[i] = 0.25
        
        # Short: Price breaks below 1d Donchian low + volatility filter
        elif (close[i] < donch_low_aligned[i] and 
              vol_filter[i]):
            signals[i] = -0.25
        
        # Exit: price returns to middle of Donchian channel
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and close[i] < (donch_high_aligned[i] + donch_low_aligned[i]) / 2) or
               (signals[i-1] == -0.25 and close[i] > (donch_high_aligned[i] + donch_low_aligned[i]) / 2))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_1d_Donchian20_VolFilter"
timeframe = "6h"
leverage = 1.0