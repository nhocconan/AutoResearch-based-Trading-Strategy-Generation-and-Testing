#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel (20) for signal direction and 1d EMA34/55 for trend filter, with volume confirmation and session filter (08-20 UTC)
# Long when price breaks above 4h Donchian upper AND 1d EMA34 > EMA55 AND volume > 2.0 * avg_volume(20) AND hour in [8,20) UTC
# Short when price breaks below 4h Donchian lower AND 1d EMA34 < EMA55 AND volume > 2.0 * avg_volume(20) AND hour in [8,20) UTC
# Exit when price crosses 4h EMA21 (trend reversal on 4h)
# Uses discrete sizing 0.20 to minimize fee churn and control drawdown
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe
# 4h Donchian provides structure with clear breakout levels
# 1d EMA34/EMA55 ensures alignment with daily trend (works in bull/bear)
# Volume confirmation filters weak breakouts
# Session filter reduces noise during low-liquidity hours

name = "1h_4hDonchian20_1dEMA34Trend_Volume_Session"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC) ONCE before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours < 20)
    
    # Get 4h data ONCE before loop for Donchian channels and EMA21
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:  # Need sufficient data for EMA21
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    donchian_upper_20 = high_series_4h.rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = low_series_4h.rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h EMA21 for exit signal
    close_series_4h = pd.Series(close_4h)
    ema_21_4h = close_series_4h.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 4h indicators to 1h timeframe (wait for completed 4h bar)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_20)
    ema_21_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Get 1d data ONCE before loop for EMA34/EMA55 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 55:  # Need sufficient data for EMA55
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 and EMA55 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_55_1d = close_series_1d.ewm(span=55, adjust=False, min_periods=55).mean().values
    
    # Align 1d indicators to 1h timeframe (wait for completed 1d bar)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_55_aligned = align_htf_to_ltf(prices, df_1d, ema_55_1d)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(ema_55_aligned[i]) or 
            np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check session filter
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 4h Donchian upper with 1d EMA34 > EMA55 and volume confirmation
            if (close[i] > donchian_upper_aligned[i] and close[i-1] <= donchian_upper_aligned[i-1] and 
                ema_34_aligned[i] > ema_55_aligned[i] and volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 4h Donchian lower with 1d EMA34 < EMA55 and volume confirmation
            elif (close[i] < donchian_lower_aligned[i] and close[i-1] >= donchian_lower_aligned[i-1] and 
                  ema_34_aligned[i] < ema_55_aligned[i] and volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price crosses below 4h EMA21 (trend reversal)
            if close[i] < ema_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price crosses above 4h EMA21 (trend reversal)
            if close[i] > ema_21_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals