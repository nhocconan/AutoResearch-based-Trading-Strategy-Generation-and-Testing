#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 1d Donchian upper band AND price > 1w EMA50 AND volume > 2.0 * avg_volume(20) on 12h
# Short when price breaks below 1d Donchian lower band AND price < 1w EMA50 AND volume > 2.0 * avg_volume(20) on 12h
# Exit when price crosses back below/above 1d Donchian middle band (mean of upper/lower)
# Uses discrete sizing 0.25 to balance return and drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# 1d Donchian provides robust structure from higher timeframe
# 1w EMA50 filters primary trend to avoid counter-trend trades in bear markets
# Volume confirmation ensures breakout strength and reduces false signals
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "12h_Donchian20_1wEMA50_VolumeSpike"
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
    
    # Get 1d data ONCE before loop for Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need at least one completed 1d bar for Donchian
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Donchian channels (based on previous 1d bar)
    # Upper = max(high_1d, lookback=20), Lower = min(low_1d, lookback=20)
    # Middle = (Upper + Lower) / 2
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Align 1d Donchian levels to 12h timeframe (wait for completed 1d bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_1d, donchian_middle)
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
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
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 1d Donchian upper, above 1w EMA50, volume confirmation, in session
            if close[i] > donchian_upper_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 1d Donchian lower, below 1w EMA50, volume confirmation, in session
            elif close[i] < donchian_lower_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 1d Donchian middle OR volume drops below average
            if close[i] < donchian_middle_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above 1d Donchian middle OR volume drops below average
            if close[i] > donchian_middle_aligned[i] or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals