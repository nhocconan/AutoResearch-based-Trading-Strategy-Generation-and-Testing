#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d EMA34 for trend direction
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 1d volume MA20 for volume filter
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 4h Donchian(20) for entry
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    upper = np.zeros(len(high_4h))
    lower = np.zeros(len(low_4h))
    for i in range(20, len(high_4h)):
        upper[i] = np.max(high_4h[i-20:i])
        lower[i] = np.min(low_4h[i-20:i])
    upper[:20] = np.nan
    lower[:20] = np.nan
    donch_upper_4h = upper
    donch_lower_4h = lower
    donch_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_upper_4h)
    donch_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_lower_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need EMA, volume, and Donchian
    start_idx = 34  # EMA34 needs 34 periods
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(donch_upper_4h_aligned[i]) or np.isnan(donch_lower_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        ema_trend = ema_34_1d_aligned[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        vol_now = volume[i]
        upper = donch_upper_4h_aligned[i]
        lower = donch_lower_4h_aligned[i]
        
        # Volume filter: volume > 1.3x 1d MA
        vol_filter = vol_now > 1.3 * vol_ma
        
        # Entry conditions: breakout with volume and trend filter
        if position == 0:
            # Long: price > EMA34 (uptrend) + break above upper band + volume
            if close[i] > ema_trend and close[i] > upper and vol_filter:
                signals[i] = size
                position = 1
            # Short: price < EMA34 (downtrend) + break below lower band + volume
            elif close[i] < ema_trend and close[i] < lower and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below EMA34 or volatility drops
            if close[i] < ema_trend or vol_now < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above EMA34 or volatility drops
            if close[i] > ema_trend or vol_now < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_EMA34_Donchian20_VolumeFilter"
timeframe = "4h"
leverage = 1.0