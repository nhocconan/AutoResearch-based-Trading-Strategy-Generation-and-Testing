#!/usr/bin/env python3
"""
Hypothesis: 1D Weekly Donchian(20) Breakout with Volume Spike and Trend Filter.
Long when price breaks above 1-week Donchian high with bullish weekly trend and volume spike.
Short when price breaks below 1-week Donchian low with bearish weekly trend and volume spike.
Exit when price returns to 1-week Donchian midline (average of high/low over 20).
Uses weekly EMA34 for trend filter to capture long-term trend and avoid whipsaws.
Designed for low trade frequency (15-25/year) to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data for Donchian channels and trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 35:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Donchian high: max of last 20 weekly highs
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    # Donchian low: min of last 20 weekly lows
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    # Donchian midline: average of high and low
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_weekly, donchian_mid)
    
    # Calculate weekly EMA34 for trend filter
    close_weekly = pd.Series(df_weekly['close'].values)
    ema34_weekly = close_weekly.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after EMA lookback
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high with bullish trend and volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema34_aligned[i] and  # Bullish trend: price above weekly EMA34
                volume[i] > 2.0 * vol_avg_20[i]):  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low with bearish trend and volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema34_aligned[i] and  # Bearish trend: price below weekly EMA34
                  volume[i] > 2.0 * vol_avg_20[i]):  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to weekly Donchian midline
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to midline
                if close[i] <= donchian_mid_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to midline
                if close[i] >= donchian_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_WeeklyDonchian_20_EMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0
#%%