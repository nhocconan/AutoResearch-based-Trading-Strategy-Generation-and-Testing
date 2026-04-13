#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA13 trend filter and volume confirmation.
# Long: price > Alligator Jaw + price > Alligator Teeth + price > EMA13 + volume > 1.5x avg volume
# Short: price < Alligator Jaw + price < Alligator Teeth + price < EMA13 + volume > 1.5x avg volume
# Alligator lines: SMA(13,8), SMA(8,5), SMA(5,3) on median price
# Trend filter: only take longs when price > EMA13, shorts when price < EMA13
# Volume confirmation reduces false signals
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Works in both bull and bear markets by using EMA13 as trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Median price for Alligator
    median_price = (high + low) / 2
    
    # 12-hour data for Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    median_price_12h = (high_12h + low_12h) / 2
    
    # Williams Alligator lines (13,8,5 periods)
    jaw = pd.Series(median_price_12h).rolling(window=13, min_periods=13).mean().values  # Blue line (13)
    teeth = pd.Series(median_price_12h).rolling(window=8, min_periods=8).mean().values    # Red line (8)
    lips = pd.Series(median_price_12h).rolling(window=5, min_periods=5).mean().values    # Green line (5)
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # 1-day EMA13 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align EMA13 to 12h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(ema_13_1d_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        ema_trend = ema_13_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: price > Jaw > Teeth > Lips + price > EMA13 + volume confirmation
            if (price > jaw_val and jaw_val > teeth_val and teeth_val > lips_val and
                price > ema_trend and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: price < Jaw < Teeth < Lips + price < EMA13 + volume confirmation
            elif (price < jaw_val and jaw_val < teeth_val and teeth_val < lips_val and
                  price < ema_trend and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price < Jaw or price < EMA13
            if (price < jaw_val or
                price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price > Jaw or price > EMA13
            if (price > jaw_val or
                price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Williams_Alligator_EMA13_Volume"
timeframe = "12h"
leverage = 1.0