#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Donchian channel breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when price breaks above daily Donchian upper (20) AND price > 1d EMA34 AND volume > 1.5 * avg_volume(20) on 12h
# Short when price breaks below daily Donchian lower (20) AND price < 1d EMA34 AND volume > 1.5 * avg_volume(20) on 12h
# Exit when price crosses back below/above daily Donchian midpoint OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Daily Donchian provides robust structure from higher timeframe
# 1d EMA34 filters primary trend to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "12h_Donchian20_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop for Donchian and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough for Donchian(20)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Donchian channels (based on previous daily bar)
    # Upper = highest high of last 20 days, Lower = lowest low of last 20 days
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_20
    donchian_lower = low_20
    donchian_mid = (donchian_upper + donchian_lower) / 2.0  # Midpoint for exit
    
    # Align daily Donchian levels to 12h timeframe (wait for completed daily bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Calculate 1d EMA34
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above daily Donchian upper, above 1d EMA34, volume confirmation, in session
            if close[i] > donchian_upper_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily Donchian lower, below 1d EMA34, volume confirmation, in session
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below daily Donchian midpoint OR volume drops below average
            if close[i] < donchian_mid_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above daily Donchian midpoint OR volume drops below average
            if close[i] > donchian_mid_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals