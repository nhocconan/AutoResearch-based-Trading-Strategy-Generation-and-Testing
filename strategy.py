#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 14-day lookback + volume confirmation + ADX filter
# Williams %R identifies overbought/oversold conditions; values below -80 indicate oversold (long setup),
# values above -20 indicate overbought (short setup). Combined with volume confirmation (>1.5x 20-bar median)
# and ADX > 25 to ensure trending markets, this strategy aims to capture mean reversion within trends.
# Works in both bull (buy oversold dips) and bear (sell overbought rallies) markets.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # 1-day ADX(14) for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean()
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean()
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean()
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean()
    adx_14 = adx.values
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Volume confirmation: current > 1.5x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 1.5 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(14, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(adx_14_aligned[i]) or 
            np.isnan(vol_threshold[i])):
            continue
        
        # ADX filter: only trade in trending markets (ADX > 25)
        if adx_14_aligned[i] <= 25:
            signals[i] = signals[i-1] if i > 0 else 0.0
            continue
        
        # Long: Williams %R oversold (< -80) + volume spike
        if (williams_r[i] < -80 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Williams %R overbought (> -20) + volume spike
        elif (williams_r[i] > -20 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Williams %R returns to neutral range (-80 to -20)
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and williams_r[i] > -80) or
               (signals[i-1] == -0.25 and williams_r[i] < -20))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_WilliamsR_Volume_ADXFilter"
timeframe = "4h"
leverage = 1.0