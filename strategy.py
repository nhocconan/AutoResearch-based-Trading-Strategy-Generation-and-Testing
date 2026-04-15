#!/usr/bin/env python3
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
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h EMA21 for trend filter
    ema_21_4h = pd.Series(df_4h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1d Donchian channels (20-period) for structure
    highest_20_1d = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    lowest_20_1d = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    highest_20_1d_aligned = align_htf_to_ltf(prices, df_1d, highest_20_1d)
    lowest_20_1d_aligned = align_htf_to_ltf(prices, df_1d, lowest_20_1d)
    
    # 1h volume ratio (current vs 20-period average) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / (vol_ma_20 + 1e-10)
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_21_4h_aligned[i]) or np.isnan(highest_20_1d_aligned[i]) or 
            np.isnan(lowest_20_1d_aligned[i]) or np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Entry conditions:
        # Long: 4h uptrend + price breaks above 1d Donchian high + volume confirmation
        # Short: 4h downtrend + price breaks below 1d Donchian low + volume confirmation
        
        if (close[i] > ema_21_4h_aligned[i] and          # 4h uptrend filter
            close[i] > highest_20_1d_aligned[i] and      # Break above 1d Donchian high
            volume_ratio[i] > 1.5):                      # Volume confirmation
            signals[i] = 0.20
            
        elif (close[i] < ema_21_4h_aligned[i] and        # 4h downtrend filter
              close[i] < lowest_20_1d_aligned[i] and     # Break below 1d Donchian low
              volume_ratio[i] > 1.5):                    # Volume confirmation
            signals[i] = -0.20
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4hEMA21_1dDonchian_Volume_SessionFilter"
timeframe = "1h"
leverage = 1.0