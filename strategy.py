#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d EMA trend filter and 12h Donchian channel breakout with volume confirmation.
# 1d EMA(50) determines trend direction (price above/below) to avoid counter-trend trades.
# 12h Donchian(20) breakout provides entry in direction of 1d trend.
# Volume confirmation (>1.3x 20-period average) reduces false breakouts.
# Exit when price returns to opposite Donchian band or trend reverses.
# Designed for low trade frequency (15-25/year) to minimize fee drag in 12h timeframe.
# Works in both bull and bear markets by using 1d trend filter to align with higher timeframe bias.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d close
    close_1d = df_1d['close'].values
    ema_period = 50
    ema_1d = pd.Series(close_1d).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Load 12h data ONCE for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    donchian_period = 20
    upper_channel = pd.Series(high_12h).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low_12h).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Align indicators to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    upper_channel_aligned = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_12h, lower_channel)
    
    # Volume confirmation: 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20, 20)  # Need EMA, Donchian, and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_1d_aligned[i]) or 
            np.isnan(upper_channel_aligned[i]) or
            np.isnan(lower_channel_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        # Trend filter: price relative to 1d EMA
        price_above_ema = close[i] > ema_1d_aligned[i]
        price_below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 0:
            # Look for Donchian breakouts in direction of 1d trend
            # Long: price breaks above upper Donchian AND price above 1d EMA (uptrend)
            if (close[i] > upper_channel_aligned[i] and 
                price_above_ema and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower Donchian AND price below 1d EMA (downtrend)
            elif (close[i] < lower_channel_aligned[i] and 
                  price_below_ema and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to lower Donchian or trend reverses (price below EMA)
            if (close[i] < lower_channel_aligned[i] or 
                price_below_ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to upper Donchian or trend reverses (price above EMA)
            if (close[i] > upper_channel_aligned[i] or 
                price_above_ema):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dEMA_12hDonchian_Breakout_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0