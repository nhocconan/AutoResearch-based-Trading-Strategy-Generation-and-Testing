#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for volume analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h data for Donchian channel and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Daily volume and 20-period average (for volume spike detection)
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # 12h Donchian channel (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high_12h = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low_12h = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA20 for trend filter
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all data to 6h timeframe
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    donchian_high_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_12h)
    donchian_low_12h_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_12h)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(volume_ma_20_1d_aligned[i]) or 
            np.isnan(donchian_high_12h_aligned[i]) or 
            np.isnan(donchian_low_12h_aligned[i]) or 
            np.isnan(ema_20_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 6h volume > 2.0x daily volume MA (adjusted for 6h)
        # 4x 6h periods per day, so daily MA/4 = approximate 6h period MA
        volume_6h_approx_ma = volume_ma_20_1d_aligned[i] / 4
        volume_condition = volume[i] > (volume_6h_approx_ma * 2.0)
        
        # Trend filter: 12h EMA20 direction
        long_trend = close[i] > ema_20_12h_aligned[i]
        short_trend = close[i] < ema_20_12h_aligned[i]
        
        # Entry conditions: Donchian breakout with volume and trend confirmation
        breakout_long = close[i] > donchian_high_12h_aligned[i]
        breakout_short = close[i] < donchian_low_12h_aligned[i]
        
        if position == 0:
            if breakout_long and volume_condition and long_trend:
                position = 1
                signals[i] = position_size
            elif breakout_short and volume_condition and short_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price breaks below 12h Donchian low or trend reverses
            if close[i] < donchian_low_12h_aligned[i] or not long_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price breaks above 12h Donchian high or trend reverses
            if close[i] > donchian_high_12h_aligned[i] or not short_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_12h1d_Donchian_Breakout_Volume_Trend_Filter_v1"
timeframe = "6h"
leverage = 1.0