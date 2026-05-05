#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) for direction and 1d EMA34 for trend filter
# Long when price breaks above 4h Donchian upper(20) AND price > 1d EMA34 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 4h Donchian lower(20) AND price < 1d EMA34 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses 4h Donchian midpoint OR volume < avg_volume(20)
# Uses discrete sizing 0.20 to minimize fee churn
# Target: 60-150 total trades over 4 years (15-37/year)
# 4h Donchian provides robust structure; 1d EMA34 filters primary trend; volume confirms breakout strength
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)
# Session filter 08-20 UTC reduces noise

name = "1h_Donchian20_4hDir_1dEMA34_VolumeSpike"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_4h_series = pd.Series(high_4h)
    low_4h_series = pd.Series(low_4h)
    donchian_upper_4h = high_4h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = low_4h_series.rolling(window=20, min_periods=20).min().values
    donchian_mid_4h = (donchian_upper_4h + donchian_lower_4h) / 2
    
    # Align 4h Donchian levels to 1h timeframe
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    donchian_mid_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid_4h)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(avg_volume_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 4h Donchian upper, above 1d EMA34, volume confirmation
            if close[i] > donchian_upper_4h_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below 4h Donchian lower, below 1d EMA34, volume confirmation
            elif close[i] < donchian_lower_4h_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 4h Donchian midpoint OR volume drops below average
            if close[i] < donchian_mid_4h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price crosses above 4h Donchian midpoint OR volume drops below average
            if close[i] > donchian_mid_4h_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals