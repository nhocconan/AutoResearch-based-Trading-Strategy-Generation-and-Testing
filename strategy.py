#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Breakout with 1d ADX Filter and Volume Spike
# Uses 12h Donchian(20) breakout as primary signal, confirmed by 1d ADX > 25 (trending market) and volume > 2x median.
# Designed to work in both bull and bear markets by capturing strong trends while avoiding chop.
# Target: 12-37 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels on 12h data
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe (no extra delay needed for breakout)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and Directional Movement
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Pad arrays to match length
    tr = np.concatenate([[0], tr])
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Calculate ATR and DI
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di_1d = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di_1d = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = pd.Series(dx_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 12h timeframe (no extra delay needed for trend strength)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Volume confirmation: current > 2x median of last 20 bars
    vol_median = pd.Series(volume).rolling(window=20, min_periods=1).median()
    vol_threshold = 2.0 * vol_median
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            continue
        
        # Long: Price breaks above 12h Donchian high, ADX > 25, volume spike
        if (close[i] > donchian_high_aligned[i] and 
            adx_1d_aligned[i] > 25 and 
            volume[i] > vol_threshold[i]):
            signals[i] = 0.25
        
        # Short: Price breaks below 12h Donchian low, ADX > 25, volume spike
        elif (close[i] < donchian_low_aligned[i] and 
              adx_1d_aligned[i] > 25 and 
              volume[i] > vol_threshold[i]):
            signals[i] = -0.25
        
        # Exit: Price returns to middle of Donchian channel or ADX weakens
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and 
                (close[i] <= (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2 or
                 adx_1d_aligned[i] <= 25)) or
               (signals[i-1] == -0.25 and 
                (close[i] >= (donchian_high_aligned[i] + donchian_low_aligned[i]) / 2 or
                 adx_1d_aligned[i] <= 25)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_Breakout_ADX_Volume"
timeframe = "12h"
leverage = 1.0