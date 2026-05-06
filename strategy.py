#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band AND 1d EMA34 rising AND volume > 1.5 * avg_volume(20) on 1h
# Short when price breaks below 4h Donchian lower band AND 1d EMA34 falling AND volume > 1.5 * avg_volume(20) on 1h
# Exit when price crosses back through the 4h Donchian midpoint (upper+lower)/2
# Uses discrete sizing 0.20 to balance return and risk
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# 4h Donchian provides strong breakout levels that reduce whipsaw
# 1d EMA34 trend filter ensures we trade with the dominant daily trend
# Volume confirmation (1.5x) validates breakout strength while limiting overtrading
# Session filter (08-20 UTC) reduces noise trades

name = "1h_4hDonchian20_1dEMA34_Trend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data ONCE before loop for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # Need at least 20 completed 4h bars for Donchian
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Donchian channels (20-period)
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    # Middle band = (upper + lower) / 2
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    donchian_upper_4h = high_series_4h.rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = low_series_4h.rolling(window=20, min_periods=20).min().values
    donchian_middle_4h = (donchian_upper_4h + donchian_lower_4h) / 2.0
    
    # Align 4h Donchian to 1h timeframe (wait for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    donchian_middle_aligned = align_htf_to_ltf(prices, df_4h, donchian_middle_4h)
    
    # Get 1d data ONCE before loop for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need at least 34 completed daily bars for EMA34
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 1h
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
            np.isnan(donchian_middle_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper band, 1d EMA34 rising, volume confirmation, in session
            if (close[i] > donchian_upper_aligned[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower band, 1d EMA34 falling, volume confirmation, in session
            elif (close[i] < donchian_lower_aligned[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses back below 4h Donchian middle band
            if close[i] < donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses back above 4h Donchian middle band
            if close[i] > donchian_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals