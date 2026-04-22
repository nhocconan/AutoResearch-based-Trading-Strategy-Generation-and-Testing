#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1d ATR volatility filter and volume confirmation
    # Works in both bull and bear markets: breakouts capture directional moves
    # ATR filter ensures breakouts occur during elevated volatility (avoid false breakouts in low vol)
    # Volume surge confirms breakout strength
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily ATR(14) for volatility filter
    tr1 = high_1d[1:] - low_1d[:-1]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_avg = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    atr_1d_ratio = atr_1d / atr_1d_avg
    atr_1d_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_1d_ratio)
    
    # 12h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper and lower bands
    upper_channel = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (20-period MA surge)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.8 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(atr_1d_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper channel with volume surge AND elevated volatility
            if close[i] > upper_channel[i] and vol_surge[i] and atr_1d_ratio_aligned[i] > 1.2:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below lower channel with volume surge AND elevated volatility
            elif close[i] < lower_channel[i] and vol_surge[i] and atr_1d_ratio_aligned[i] > 1.2:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian channel
            if position == 1:
                if close[i] < lower_channel[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper_channel[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dATR_VolumeSurge_v1"
timeframe = "12h"
leverage = 1.0