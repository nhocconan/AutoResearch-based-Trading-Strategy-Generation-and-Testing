#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d ADX trend filter and volume confirmation.
# Works in bull/bear by only trading in the direction of higher timeframe trend.
# Targets 15-25 trades/year (60-100 total) with low frequency to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    # Load 1d data once for ADX and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-period ADX for trend strength
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), np.abs(high_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], tr1, tr2])
    tr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    plus_dm = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    plus_di = 100 * plus_dm / tr
    minus_di = 100 * minus_dm / tr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 6h Donchian channels (20-period)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    donch_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Volume spike filter
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if any data is not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        adx_val = adx_aligned[i]
        
        # Volume filter: current volume > 1.8 * 20-day average
        vol_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Long: price breaks above Donchian high + ADX > 25 + volume spike
            if price > donch_high[i] and adx_val > 25 and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + ADX > 25 + volume spike
            elif price < donch_low[i] and adx_val > 25 and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through Donchian middle or ADX weakens
            donch_mid = (donch_high[i] + donch_low[i]) / 2
            exit_signal = False
            
            if position == 1:  # long position
                if price < donch_mid or adx_val < 20:
                    exit_signal = True
            elif position == -1:  # short position
                if price > donch_mid or adx_val < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian20_ADX25_Volume"
timeframe = "6h"
leverage = 1.0