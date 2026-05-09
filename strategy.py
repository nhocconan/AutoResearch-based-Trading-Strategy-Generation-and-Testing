#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_1dTrend_Volume_Spike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter and volume spike detection
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume spike: current volume > 1.5 * 20-period average
    vol_series_1d = pd.Series(volume_1d)
    vol_ma_20_1d = vol_series_1d.rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (vol_ma_20_1d * 1.5)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # 12h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        dh = donchian_high[i]
        dl = donchian_low[i]
        ema_trend = ema_34_1d_aligned[i]
        vol_spike = volume_spike_1d_aligned[i]
        
        if position == 0:
            # Enter long: Price breaks above Donchian high + uptrend + volume spike
            if close[i] > dh and close[i] > ema_trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Donchian low + downtrend + volume spike
            elif close[i] < dl and close[i] < ema_trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below Donchian low
            if close[i] < dl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above Donchian high
            if close[i] > dh:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals