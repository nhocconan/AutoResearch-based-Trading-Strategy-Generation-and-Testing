#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian20_1dTrend_Volume_v1"
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
    
    # Get 1d data for trend filter and volume baseline
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume average for spike detection
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need EMA34 and Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_avg_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_trend = ema34_1d_aligned[i]
        vol_avg = vol_avg_1d_aligned[i]
        vol_spike = volume[i] > (vol_avg * 2.0)
        in_session = session_mask[i]
        upper_channel = donchian_high[i]
        lower_channel = donchian_low[i]
        
        if position == 0:
            # Enter long: Break above Donchian high + 1d uptrend + volume spike + session
            if close[i] > upper_channel and close[i] > ema_trend and vol_spike and in_session:
                signals[i] = 0.25
                position = 1
            # Enter short: Break below Donchian low + 1d downtrend + volume spike + session
            elif close[i] < lower_channel and close[i] < ema_trend and vol_spike and in_session:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close falls below Donchian low or 1d trend turns down
            if close[i] < lower_channel or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close rises above Donchian high or 1d trend turns up
            if close[i] > upper_channel or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals