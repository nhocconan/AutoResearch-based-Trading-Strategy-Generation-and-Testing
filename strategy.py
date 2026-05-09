#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 4h data for Donchian channels and volume
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Donchian channels (20-period) on 4h data
    high_series = pd.Series(df_4h['high'])
    low_series = pd.Series(df_4h['low'])
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 4h volume > 1.8 * 20-period average
    vol_series = pd.Series(df_4h['volume'])
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = df_4h['volume'] > (vol_ma * 1.8)
    
    # Align all to 4h timeframe
    donchian_high_4h = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_4h = align_htf_to_ltf(prices, df_4h, donchian_low)
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_filter_4h = align_htf_to_ltf(prices, df_4h, volume_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_4h[i]) or np.isnan(donchian_low_4h[i]) or
            np.isnan(ema50_1d_4h[i]) or np.isnan(volume_filter_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = donchian_high_4h[i]
        lower = donchian_low_4h[i]
        trend = ema50_1d_4h[i]
        vol_ok = volume_filter_4h[i]
        
        if position == 0:
            # Enter long: break above upper Donchian with volume and above trend
            if close[i] > upper and close[i] > trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower Donchian with volume and below trend
            elif close[i] < lower and close[i] < trend and vol_ok:
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