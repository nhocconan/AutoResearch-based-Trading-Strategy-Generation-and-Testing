#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mpf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d HTF for trend and volatility filtering.
# Uses 12h Donchian channel breakouts confirmed by 1d EMA trend and 12h volume spike.
# Designed to work in both bull and bear markets by using volatility-adjusted breakouts
# and trend filtering to avoid whipsaws. Target: 15-30 trades/year to minimize fee drag.

name = "12h_Donchian_20_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 12h data for Donchian channels and volume
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_series = pd.Series(df_12h['high'].values)
    low_series = pd.Series(df_12h['low'].values)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current 12h volume > 2.0 * 20-period average
    vol_series = pd.Series(df_12h['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = df_12h['volume'].values > (vol_ma * 2.0)
    
    # Align all indicators to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema50_1d_aligned[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        vol_spike = volume_spike_aligned[i]
        
        if position == 0:
            # Enter long: break above upper Donchian with uptrend and volume spike
            if close[i] > upper and close[i] > trend and vol_spike:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower Donchian with downtrend and volume spike
            elif close[i] < lower and close[i] < trend and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below lower Donchian (mean reversion)
            if close[i] < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above upper Donchian (mean reversion)
            if close[i] > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals