#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian breakout with 1d ATR filter and volume surge
    # Works in bull and bear markets: breakouts capture directional moves, ATR filters volatility regime
    # Volume surge confirms breakout strength, reducing false signals
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily ATR(14) for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - low_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - high_1d[:-1]))
    tr = np.concatenate([[np.nan], tr1])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 12h Donchian channels (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate Donchian upper and lower bands
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter (20-period MA surge)
    vol_ma20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_surge = prices['volume'].values > 1.8 * vol_ma20
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma20[i]) or np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Donchian breakout above upper band with volume surge AND ATR above median (volatile market)
            if close[i] > upper[i] and vol_surge[i] and atr_1d_aligned[i] > np.nanmedian(atr_1d_aligned[max(0, i-100):i+1]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band with volume surge AND ATR above median
            elif close[i] < lower[i] and vol_surge[i] and atr_1d_aligned[i] > np.nanmedian(atr_1d_aligned[max(0, i-100):i+1]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian band or ATR drops below median (low volatility)
            if position == 1:
                if close[i] < lower[i] or atr_1d_aligned[i] < np.nanmedian(atr_1d_aligned[max(0, i-100):i+1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > upper[i] or atr_1d_aligned[i] < np.nanmedian(atr_1d_aligned[max(0, i-100):i+1]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_1dATR_VolumeSurge_v1"
timeframe = "12h"
leverage = 1.0