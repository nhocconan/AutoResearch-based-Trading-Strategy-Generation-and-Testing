#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for Donchian breakout
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Daily data for volume and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period high/low) on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 20-period highest high and lowest low
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h EMA20 for trend filter
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 4h volume > 1.5x 20-period average daily volume (scaled)
        # Approximate: 6x 4h periods per day, so daily MA/6 = 4h period MA
        volume_4h_approx_ma = volume_ma_20_1d_aligned[i] / 6
        volume_condition = volume[i] > (volume_4h_approx_ma * 1.5)
        
        # Trend filter: 12h EMA20 direction
        # Long when price > 12h EMA20
        # Short when price < 12h EMA20
        long_trend = close[i] > ema_20_12h_aligned[i]
        short_trend = close[i] < ema_20_12h_aligned[i]
        
        # Entry conditions: Donchian breakout with volume and trend confirmation
        # Long when price breaks above Donchian high with volume and uptrend
        # Short when price breaks below Donchian low with volume and downtrend
        breakout_long = close[i] > donchian_high_aligned[i]
        breakout_short = close[i] < donchian_low_aligned[i]
        
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
            # Exit when price breaks below Donchian low or shows reversal
            if close[i] < donchian_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price breaks above Donchian high or shows reversal
            if close[i] > donchian_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h1d_Donchian_Breakout_Volume_Trend_Filter_v1"
timeframe = "4h"
leverage = 1.0