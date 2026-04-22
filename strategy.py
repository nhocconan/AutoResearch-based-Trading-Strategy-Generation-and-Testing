#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # 1d Close for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily Donchian(20) - upper and lower bands
    high_max_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    upper_band = align_htf_to_ltf(prices, df_1d, high_max_20)
    lower_band = align_htf_to_ltf(prices, df_1d, low_min_20)
    
    # 12h ATR for volatility filter
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 12h Volume confirmation
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.8 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian band + volume surge + ATR filter
            if (close[i] > upper_band[i] and vol_surge[i] and atr[i] > 0):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian band + volume surge + ATR filter
            elif (close[i] < lower_band[i] and vol_surge[i] and atr[i] > 0):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle of Donchian channel
            mid_band = (upper_band[i] + lower_band[i]) / 2
            if position == 1:
                if close[i] < mid_band:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > mid_band:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_DonchianBreakout_VolumeSurge_ATRFilter_v1"
timeframe = "12h"
leverage = 1.0