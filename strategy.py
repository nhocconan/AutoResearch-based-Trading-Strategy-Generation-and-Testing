#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_1dTrend_Volume"
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
    
    # Get 1d data for Donchian, trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Donchian(20) from 1d high/low
    high_series = pd.Series(df_1d['high'])
    low_series = pd.Series(df_1d['low'])
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current 1d volume > 1.5 * 20-period average
    vol_series = pd.Series(df_1d['volume'])
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = df_1d['volume'].values > (vol_ma * 1.5)
    
    # Align all to 4h
    dh_4h = align_htf_to_ltf(prices, df_1d, donchian_high)
    dl_4h = align_htf_to_ltf(prices, df_1d, donchian_low)
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_filter_4h = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Need enough data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(dh_4h[i]) or np.isnan(dl_4h[i]) or
            np.isnan(ema50_1d_4h[i]) or np.isnan(vol_filter_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        dh_val = dh_4h[i]
        dl_val = dl_4h[i]
        trend = ema50_1d_4h[i]
        vol_ok = vol_filter_4h[i]
        
        if position == 0:
            # Enter long: break above Donchian high with volume and above trend
            if close[i] > dh_val and close[i] > trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: break below Donchian low with volume and below trend
            elif close[i] < dl_val and close[i] < trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below Donchian low (mean reversion)
            if close[i] < dl_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above Donchian high (mean reversion)
            if close[i] > dh_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals