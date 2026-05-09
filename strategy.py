#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_1dTrend_Volume_Breakout"
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
    
    # Get 1d data for trend and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Donchian channel (20-period) on 4h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 1d volume filter: current volume > 1.5 * 20-period average
    vol_1d_series = pd.Series(df_1d['volume'])
    vol_ma_1d = vol_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_filter_1d = df_1d['volume'].values > (vol_ma_1d * 1.5)
    
    # Align 1d indicators to 4h timeframe
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_filter_1d_4h = align_htf_to_ltf(prices, df_1d, vol_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, 20)  # Need enough data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_1d_4h[i]) or np.isnan(vol_filter_1d_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        dh = donchian_high[i]
        dl = donchian_low[i]
        trend = ema50_1d_4h[i]
        vol_ok = vol_filter_1d_4h[i]
        
        if position == 0:
            # Enter long: break above Donchian high with volume and uptrend
            if close[i] > dh and close[i] > trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: break below Donchian low with volume and downtrend
            elif close[i] < dl and close[i] < trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below Donchian low (mean reversion)
            if close[i] < dl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above Donchian high (mean reversion)
            if close[i] > dh:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals