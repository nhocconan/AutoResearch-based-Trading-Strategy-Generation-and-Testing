#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_VolumeSpike_1dTrend_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA100 for trend filter
    ema100_1d = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # 4h Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Align 1d EMA100 to 4h
    ema100_1d_4h = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema100_1d_4h[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema100_1d_4h[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: break above Donchian high with volume spike and above trend
            if close[i] > upper and close[i] > trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: break below Donchian low with volume spike and below trend
            elif close[i] < lower and close[i] < trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below Donchian low (mean reversion)
            if close[i] < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above Donchian high (mean reversion)
            if close[i] > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals