#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian breakout with weekly EMA trend filter and volume confirmation
# Uses 1d Donchian(20) breakouts in direction of 1-week EMA(34) trend
# Volume spike confirms breakout strength. Target: 15-25 trades/year per symbol (60-100 total)
# Weekly trend filter avoids counter-trend losses in choppy markets, reducing false breakouts
# Focus on major trend moves in both bull and bear markets with controlled risk

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for Donchian channels and volume calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on daily data
    # Upper band: 20-day high, Lower band: 20-day low
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily volume average for spike detection
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Load weekly data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 34-period EMA on weekly close for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align indicators to daily timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(vol_avg_20_aligned[i]) or np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-day high + volume spike + weekly uptrend
            if (close[i] > high_20_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20_aligned[i] and
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low + volume spike + weekly downtrend
            elif (close[i] < low_20_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20_aligned[i] and
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite Donchian level
            if position == 1:
                if close[i] < low_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > high_20_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyEMA34_Volume_Trend"
timeframe = "1d"
leverage = 1.0