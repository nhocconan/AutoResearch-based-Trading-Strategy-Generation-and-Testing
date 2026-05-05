#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper band AND price > 1d EMA50 AND volume > 1.5 * avg_volume(20)
# Short when price breaks below 12h Donchian lower band AND price < 1d EMA50 AND volume > 1.5 * avg_volume(20)
# Exit when price crosses back below/above 12h Donchian middle band (20-period mean)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 100-180 total trades over 4 years (25-45/year) for 4h timeframe
# 12h Donchian provides robust structure from higher timeframe
# 1d EMA50 filters primary trend to avoid counter-trend trades
# Volume confirmation reduces false signals
# Works in bull markets (breakouts with uptrend) and bear markets (breakdowns with downtrend)

name = "4h_Donchian20_12h_1dEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:  # Need enough for Donchian(20)
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_middle = (high_20 + low_20) / 2.0
    
    # Align 12h Donchian channels to 4h timeframe (wait for completed 12h bar)
    donchian_ub_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    donchian_lb_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_middle)
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_ub_aligned[i]) or np.isnan(donchian_lb_aligned[i]) or 
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 12h Donchian upper band, above 1d EMA50, volume confirmation, in session
            if close[i] > donchian_ub_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 12h Donchian lower band, below 1d EMA50, volume confirmation, in session
            elif close[i] < donchian_lb_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below 12h Donchian middle band
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above 12h Donchian middle band
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals