#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian Breakout with 1d Trend and Volume Confirmation
Long when price breaks above Donchian(20) high with bullish 1d trend and volume spike.
Short when price breaks below Donchian(20) low with bearish 1d trend and volume spike.
Exit when price touches Donchian mid-level.
Uses 1d EMA34 for trend filter to avoid whipsaws in choppy markets.
Designed for low trade frequency (12-37/year) with 12h timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for trend filter and Donchian calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 12h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate daily OHLC for Donchian channels
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    # Donchian(20) on 1d: upper band = max(high_d, 20), lower band = min(low_d, 20)
    high_series = pd.Series(high_d)
    low_series = pd.Series(low_d)
    donch_high_1d = high_series.rolling(window=20, min_periods=20).max().values
    donch_low_1d = low_series.rolling(window=20, min_periods=20).min().values
    donch_mid_1d = (donch_high_1d + donch_low_1d) / 2.0
    
    # Align Donchian levels to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high_1d)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low_1d)
    donch_mid_aligned = align_htf_to_ltf(prices, df_1d, donch_mid_1d)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after lookbacks
        # Skip if data not ready
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(donch_mid_aligned[i]) or np.isnan(ema34_aligned[i]) or 
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
            # Long: Price breaks above Donchian high with bullish 1d trend and volume spike
            if (close[i] > donch_high_aligned[i] and 
                close[i] > ema34_aligned[i] and  # Bullish trend: price above EMA34
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with bearish 1d trend and volume spike
            elif (close[i] < donch_low_aligned[i] and 
                  close[i] < ema34_aligned[i] and  # Bearish trend: price below EMA34
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to Donchian mid-level
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to Donchian mid
                if close[i] <= donch_mid_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to Donchian mid
                if close[i] >= donch_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_20_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0
#%%