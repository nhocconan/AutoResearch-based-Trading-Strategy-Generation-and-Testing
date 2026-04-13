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
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Daily data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly EMA for trend filter
    ema_20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean()
    ema_20_1w_values = ema_20_1w.values
    
    # Daily Donchian channels (20-period high/low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Daily volume and its 20-period average
    volume_1d = df_1d['volume'].values
    volume_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all data to 12h timeframe
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w_values)
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    volume_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if any required data is not ready
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(donchian_high_20_aligned[i]) or
            np.isnan(donchian_low_20_aligned[i]) or np.isnan(volume_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 12h volume > 1.5x 20-period average
        # Approximate 12h volume from daily volume (assuming 2x 12h periods per day)
        volume_12h = volume[i]  # Current 12h bar volume
        volume_ma_20_12h = volume_ma_20_1d_aligned[i] / 2  # Approximate 20-period average for 12h
        volume_condition = volume_12h > (volume_ma_20_12h * 1.5)
        
        # Trend filter: only long when price > weekly EMA20, short when price < weekly EMA20
        long_trend = close[i] > ema_20_1w_aligned[i]
        short_trend = close[i] < ema_20_1w_aligned[i]
        
        # Entry conditions: Donchian breakout with volume and trend confirmation
        # Long when price breaks above Donchian high with volume and uptrend
        # Short when price breaks below Donchian low with volume and downtrend
        if position == 0:
            if close[i] > donchian_high_20_aligned[i] and volume_condition and long_trend:
                position = 1
                signals[i] = position_size
            elif close[i] < donchian_low_20_aligned[i] and volume_condition and short_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when price returns to Donchian low or shows reversal
            if close[i] < donchian_low_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when price returns to Donchian high or shows reversal
            if close[i] > donchian_high_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1w1d_Donchian_Breakout_With_Volume_Confirmation_v1"
timeframe = "12h"
leverage = 1.0