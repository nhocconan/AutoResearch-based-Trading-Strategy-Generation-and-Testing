#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Donchian20_1dTrend_Volume_Spike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volume baseline
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 20:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_d = df_d['close'].values
    ema34_d = pd.Series(close_d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)
    
    # Daily volume baseline for spike detection
    vol_d = df_d['volume'].values
    vol_ma20_d = pd.Series(vol_d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_d_aligned = align_htf_to_ltf(prices, df_d, vol_ma20_d)
    
    # Calculate 6h Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=20, min_periods=20).max().values
    donch_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema34_d_aligned[i]) or 
            np.isnan(vol_ma20_d_aligned[i]) or
            np.isnan(donch_high[i]) or
            np.isnan(donch_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_d_aligned[i]
        vol_ma20_val = vol_ma20_d_aligned[i]
        donch_high_val = donch_high[i]
        donch_low_val = donch_low[i]
        
        # Volume spike condition: current volume > 1.8 * daily 20-period average
        vol_spike = volume[i] > (vol_ma20_val * 1.8)
        
        if position == 0:
            # Enter long: Price breaks above Donchian high + above daily EMA34 + volume spike
            if close[i] > donch_high_val and close[i] > ema34_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Donchian low + below daily EMA34 + volume spike
            elif close[i] < donch_low_val and close[i] < ema34_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below Donchian low or below daily EMA34
            if close[i] < donch_low_val or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above Donchian high or above daily EMA34
            if close[i] > donch_high_val or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals