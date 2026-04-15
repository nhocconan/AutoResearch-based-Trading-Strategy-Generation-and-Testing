#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 12h volume confirmation + 1d EMA trend filter
# Designed for low trade frequency (target 20-40/year) with clear trend-following logic
# Works in bull markets (breakouts above upper band) and bear markets (breakdowns below lower band)
# Uses Donchian channels for breakout detection, volume to confirm conviction, and EMA for trend alignment
# Conservative sizing (0.25) to manage drawdowns during choppy periods

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 12h data for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    volume_12h = df_12h['volume'].values
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) on 4h
    # Upper band = highest high over last 20 periods
    # Lower band = lowest low over last 20 periods
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    donchian_upper = high_series_4h.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series_4h.rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period on 12h)
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # EMA50 on 1d for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    vol_avg_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_12h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # Conservative position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_avg_12h_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            continue
        
        # Long entry: price breaks above Donchian upper band + volume confirmation + uptrend
        if (high[i] > donchian_upper_aligned[i] and 
            volume[i] > 1.5 * vol_avg_12h_aligned[i] and 
            close[i] > ema50_1d_aligned[i] and 
            position <= 0):
            position = 1
            signals[i] = position_size
        
        # Short entry: price breaks below Donchian lower band + volume confirmation + downtrend
        elif (low[i] < donchian_lower_aligned[i] and 
              volume[i] > 1.5 * vol_avg_12h_aligned[i] and 
              close[i] < ema50_1d_aligned[i] and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        
        # Exit: reverse signal or price returns to the middle of the channel
        elif position == 1 and close[i] <= (donchian_upper_aligned[i] + donchian_lower_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= (donchian_upper_aligned[i] + donchian_lower_aligned[i]) / 2:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_12hVolume_1dEMA_Trend"
timeframe = "4h"
leverage = 1.0