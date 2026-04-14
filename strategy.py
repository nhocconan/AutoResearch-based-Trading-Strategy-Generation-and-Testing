#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly Donchian channel breakout with 1-day RSI filter and volume confirmation.
# Weekly Donchian breakout (20-period) captures major trend changes.
# RSI(14) between 40-60 avoids overbought/oversold extremes for better entry quality.
# Volume confirmation (>1.5x 20-day average) reduces false breakouts.
# Designed to work in both bull and bear markets by capturing breakouts in either direction.
# Target: 10-20 trades/year per symbol (40-80 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate Donchian channels on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_period = 20
    upper_channel = pd.Series(high_1w).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low_1w).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Load daily data ONCE for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI on daily data
    close_1d = df_1d['close'].values
    
    rsi_period = 14
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align indicators to daily timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_1w, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1w, lower_channel)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14, 20)  # Donchian, RSI, and volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or
            np.isnan(rsi_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # RSI filter: avoid extremes (40-60 range)
        rsi_filter = (rsi_aligned[i] >= 40) & (rsi_aligned[i] <= 60)
        
        if position == 0:
            # Look for Donchian channel breakouts
            # Long: price breaks above upper weekly Donchian channel
            if (close[i] > upper_channel_aligned[i] and 
                volume_confirmed and 
                rsi_filter):
                position = 1
                signals[i] = position_size
            # Short: price breaks below lower weekly Donchian channel
            elif (close[i] < lower_channel_aligned[i] and 
                  volume_confirmed and 
                  rsi_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to weekly Donchian midpoint or opposite breakout
            mid_channel = (upper_channel_aligned[i] + lower_channel_aligned[i]) / 2
            if (close[i] <= mid_channel or 
                close[i] < lower_channel_aligned[i]):  # Opposite breakout
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to weekly Donchian midpoint or opposite breakout
            mid_channel = (upper_channel_aligned[i] + lower_channel_aligned[i]) / 2
            if (close[i] >= mid_channel or 
                close[i] > upper_channel_aligned[i]):  # Opposite breakout
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1wDonchian_RSI_VolumeFilter_v1"
timeframe = "1d"
leverage = 1.0