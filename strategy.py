#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + 1d Volume Spike + ADX Trend Filter
# Uses Donchian channel breakouts for directional entries, confirmed by 1d volume spikes.
# ADX filter ensures we only trade in trending regimes (ADX > 25) to avoid whipsaws in ranging markets.
# Designed to work in both bull and bear markets by capturing strong momentum moves.
# Conservative sizing (0.25) to limit trade frequency and avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day volume for spike detection
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean()
    vol_spike_1d = volume_1d > (2.0 * vol_ma_1d)  # 2x 20-day average volume
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    
    # ADX for trend strength (14-period)
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).mean() / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).mean() / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    signals = np.zeros(n)
    
    for i in range(40, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_spike_1d_aligned[i])):
            continue
        
        # Long: Price breaks above Donchian high, ADX > 25 (trending), volume spike
        if (close[i] > donchian_high[i-1] and 
            adx[i] > 25 and 
            vol_spike_1d_aligned[i]):
            signals[i] = 0.25
        
        # Short: Price breaks below Donchian low, ADX > 25 (trending), volume spike
        elif (close[i] < donchian_low[i-1] and 
              adx[i] > 25 and 
              vol_spike_1d_aligned[i]):
            signals[i] = -0.25
        
        # Exit: Price returns to middle of Donchian channel or ADX weakens
        elif (i > 0 and 
              ((signals[i-1] == 0.25 and (close[i] < (donchian_high[i-1] + donchian_low[i-1]) / 2 or adx[i] < 20)) or
               (signals[i-1] == -0.25 and (close[i] > (donchian_high[i-1] + donchian_low[i-1]) / 2 or adx[i] < 20)))):
            signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_Volume_ADX"
timeframe = "4h"
leverage = 1.0