#1/2025-06-20
#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation.
Long when price breaks above Donchian upper band with bullish 1d trend and volume spike.
Short when price breaks below Donchian lower band with bearish 1d trend and volume spike.
Exit when price returns to Donchian middle band or trend weakens.
Uses 1d EMA34 for trend filter to avoid whipsaws in ranging markets.
Designed for low trade frequency (12-37/year) to minimize fee drift.
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
    
    # Load daily data for trend filter - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 35:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_d = pd.Series(df_daily['close'].values)
    ema34_d = close_d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align EMA34 to 12h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_daily, ema34_d)
    
    # Calculate Donchian channels from 1d data (20-period)
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    close_d = df_daily['close'].values
    
    # Donchian upper/lower bands (20-period)
    upper_d = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    lower_d = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    middle_d = (upper_d + lower_d) / 2.0
    
    # Align Donchian bands to 12h timeframe
    upper_aligned = align_htf_to_ltf(prices, df_daily, upper_d)
    lower_aligned = align_htf_to_ltf(prices, df_daily, lower_d)
    middle_aligned = align_htf_to_ltf(prices, df_daily, middle_d)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):  # Start after lookbacks
        # Skip if data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(middle_aligned[i]) or np.isnan(ema34_aligned[i]) or 
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
            # Long: Price breaks above upper band with bullish 1d trend and volume spike
            if (close[i] > upper_aligned[i] and 
                close[i] > ema34_aligned[i] and  # Bullish trend: price above EMA34
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band with bearish 1d trend and volume spike
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema34_aligned[i] and  # Bearish trend: price below EMA34
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle band OR trend turns bearish
                if close[i] <= middle_aligned[i] or close[i] < ema34_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle band OR trend turns bullish
                if close[i] >= middle_aligned[i] or close[i] > ema34_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_UpperLower_1dEMA34_Trend_Volume"
timeframe = "12h"
leverage = 1.0
#%%