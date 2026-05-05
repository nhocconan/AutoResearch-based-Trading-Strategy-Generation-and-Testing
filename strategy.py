#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation
# Long when price breaks above daily Donchian upper (20) AND price > 1d EMA50 AND volume > 2.0 * avg_volume(20) on 12h
# Short when price breaks below daily Donchian lower (20) AND price < 1d EMA50 AND volume > 2.0 * avg_volume(20) on 12h
# Exit when price crosses back below/above daily Donchian midpoint OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 80-120 total trades over 4 years (20-30/year) for 12h timeframe
# Daily Donchian provides robust support/resistance from higher timeframe
# 1d EMA50 filters primary trend to avoid counter-trend trades
# Volume spike confirms breakout strength and reduces false signals
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "12h_Donchian20_1dEMA50_VolumeSpike"
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
    
    # Get daily data ONCE before loop for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least one completed daily bar for Donchian
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Donchian channels (based on previous daily bar)
    # Upper = max(high, lookback=20), Lower = min(low, lookback=20)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0  # Midpoint for exit
    
    # Align daily Donchian levels to 12h timeframe (wait for completed daily bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d_ema = get_htf_data(prices, '1d')  # Reuse same daily data
    if len(df_1d_ema) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1d_ema = df_1d_ema['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d_ema)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d_ema, ema50_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above daily Donchian upper, above 1d EMA50, volume confirmation, in session
            if close[i] > donchian_upper_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below daily Donchian lower, below 1d EMA50, volume confirmation, in session
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_confirm[i]:
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