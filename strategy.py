#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian(20) breakout with 1-week trend filter and volume confirmation.
Long when price breaks above 1-week high with bullish 1-week trend and volume spike.
Short when price breaks below 1-week low with bearish 1-week trend and volume spike.
Exit when price returns to 1-week midpoint.
Designed for low trade frequency (5-10/year) to minimize fee drift on 1d timeframe.
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
    
    # Load 1w data for trend filter and Donchian channels - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 25:
        return np.zeros(n)
    
    # Calculate 1-week EMA21 for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema21_1w = close_1w.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align EMA21 to daily timeframe
    ema21_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # Calculate weekly Donchian channels (20-period high/low)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Weekly high: highest high of last 20 weeks
    high_series = pd.Series(high_1w)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    
    # Weekly low: lowest low of last 20 weeks
    low_series = pd.Series(low_1w)
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly midpoint for exit
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align all weekly levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or np.isnan(ema21_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly high with bullish 1w trend and volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema21_aligned[i] and  # Bullish trend: price above weekly EMA21
                volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly low with bearish 1w trend and volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema21_aligned[i] and  # Bearish trend: price below weekly EMA21
                  volume[i] > 2.0 * vol_avg_20[i]):  # Strong volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to weekly midpoint
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to weekly midpoint
                if close[i] <= donchian_mid_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to weekly midpoint
                if close[i] >= donchian_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_20_1wEMA21_Trend_Volume"
timeframe = "1d"
leverage = 1.0
#%%