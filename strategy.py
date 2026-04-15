#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian Channel Breakout + 1d Volume Spike + ADX Trend Filter
# Uses 12h Donchian(20) breakout with volume confirmation and ADX(14) > 25 for trend strength
# Works in bull (breakouts above upper band) and bear (breakdowns below lower band)
# Discrete sizing (0.25) to limit overtrading and fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h Donchian Channel (20-period)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian bands on 12h data
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align to 12h timeframe (wait for bar close)
    donchian_high = align_htf_to_ltf(prices, df_12h, high_20)
    donchian_low = align_htf_to_ltf(prices, df_12h, low_20)
    
    # 1d Volume Spike (relative to 20-period median)
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_median_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).median().values
    vol_spike_threshold = 2.0 * vol_median_1d  # Require 2x median volume
    vol_spike_1d = align_htf_to_ltf(prices, df_1d, vol_spike_threshold)
    
    # ADX(14) for trend strength on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx[np.isnan(dx)] = 0  # Handle division by zero
    
    # Align ADX to 12h timeframe
    adx_12h = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_spike_1d[i]) or np.isnan(adx_12h[i])):
            continue
        
        # Long: Price breaks above Donchian high + volume spike + strong trend (ADX > 25)
        if (close[i] > donchian_high[i] and 
            volume[i] > vol_spike_1d[i] and 
            adx_12h[i] > 25):
            signals[i] = 0.25
        
        # Short: Price breaks below Donchian low + volume spike + strong trend (ADX > 25)
        elif (close[i] < donchian_low[i] and 
              volume[i] > vol_spike_1d[i] and 
              adx_12h[i] > 25):
            signals[i] = -0.25
        
        # Exit: Price returns to middle of Donchian channel or trend weakens
        elif i > 0:
            mid = (donchian_high[i] + donchian_low[i]) / 2
            if (signals[i-1] == 0.25 and (close[i] < mid or adx_12h[i] < 20)) or \
               (signals[i-1] == -0.25 and (close[i] > mid or adx_12h[i] < 20)):
                signals[i] = 0.0
        
        # Otherwise, hold previous position
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "12h_Donchian_Volume_ADX_Trend"
timeframe = "12h"
leverage = 1.0