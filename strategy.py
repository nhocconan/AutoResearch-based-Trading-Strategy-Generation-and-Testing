#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1-week EMA200 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band with bullish weekly trend and volume > 1.5x average.
# Short when price breaks below Donchian lower band with bearish weekly trend and volume > 1.5x average.
# Exit when price re-enters Donchian channel.
# Designed for low trade frequency (<15 per year per symbol) to minimize fee drag and improve generalization.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channel calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least 20 days for Donchian(20)
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian(20) channels from previous day's data
    # Upper band = 20-day high, Lower band = 20-day low
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian bands to 1d timeframe (already 1d, but using for consistency)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1-week EMA200 for trend filter
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Volume filter: volume > 1.5x 50-period average (moderate to balance signal quality and frequency)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60  # Need sufficient data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above Donchian upper band, above weekly EMA200, volume spike
        if (close[i] > donchian_upper_aligned[i] and 
            close[i] > ema200_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below Donchian lower band, below weekly EMA200, volume spike
        elif (close[i] < donchian_lower_aligned[i] and 
              close[i] < ema200_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price re-enters Donchian channel
        elif position == 1 and close[i] < donchian_upper_aligned[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donchian_lower_aligned[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1wEMA200_VolumeFilter"
timeframe = "1d"
leverage = 1.0