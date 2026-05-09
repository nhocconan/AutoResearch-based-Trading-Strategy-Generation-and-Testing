#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (1d EMA34)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 34:
        return np.zeros(n)
    
    # Calculate 34-period EMA on daily close
    close_d = df_d['close'].values
    ema34_d = pd.Series(close_d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)
    
    # Get 12h data for Donchian channel (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian upper/lower (20-period high/low)
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Volume spike filter: current volume > 2.0 * 20-period average (tight for 12h)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 20)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema34_d_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_d_val = ema34_d_aligned[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Enter long: Price breaks above Donchian high + daily uptrend + volume spike
            if close[i] > upper and close[i] > ema34_d_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Donchian low + daily downtrend + volume spike
            elif close[i] < lower and close[i] < ema34_d_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below Donchian low or daily trend turns down
            if close[i] < lower or close[i] < ema34_d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above Donchian high or daily trend turns up
            if close[i] > upper or close[i] > ema34_d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals