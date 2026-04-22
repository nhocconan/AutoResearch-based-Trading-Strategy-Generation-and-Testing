#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian channel breakout with weekly trend filter and volume confirmation.
Long when price breaks above upper Donchian(20) with bullish weekly trend (price > weekly EMA34).
Short when price breaks below lower Donchian(20) with bearish weekly trend (price < weekly EMA34).
Exit when price returns to opposite Donchian band or weekly trend reverses.
Designed for low trade frequency (10-25/year) to minimize fee drift and capture major trends.
Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
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
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 35:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_w = pd.Series(df_weekly['close'].values)
    ema34_w = close_w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_w_aligned = align_htf_to_ltf(prices, df_weekly, ema34_w)
    
    # Calculate daily Donchian channels (20-period)
    # We need daily high/low for the past 20 completed days
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_d = df_daily['high'].values
    low_d = df_daily['low'].values
    
    # Calculate Donchian channels on daily timeframe
    # Upper = max(high of last 20 days), Lower = min(low of last 20 days)
    upper_d = pd.Series(high_d).rolling(window=20, min_periods=20).max().values
    lower_d = pd.Series(low_d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (already aligned since we're using daily data)
    # But we need to align to our trading timeframe (1d)
    # Since we're trading on 1d, we can use the daily values directly
    # However, we need to ensure we only use completed daily bars
    # The rolling calculation already uses completed periods
    
    # For 1d timeframe, we shift by 1 to avoid look-ahead (use previous day's Donchian)
    upper_d_shifted = np.roll(upper_d, 1)
    lower_d_shifted = np.roll(lower_d, 1)
    upper_d_shifted[0] = np.nan  # First value invalid
    lower_d_shifted[0] = np.nan
    
    # Since we're trading on 1d timeframe, no further alignment needed
    # But we need to make sure arrays are same length as prices
    # We'll use the same index - prices are already 1d
    
    # Calculate volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian lookback
        # Skip if data not ready
        if (np.isnan(upper_d_shifted[i]) or np.isnan(lower_d_shifted[i]) or 
            np.isnan(ema34_w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian with bullish weekly trend and volume spike
            if (close[i] > upper_d_shifted[i] and 
                close[i] > ema34_w_aligned[i] and  # Bullish trend: price above weekly EMA34
                volume[i] > 1.5 * vol_avg_20[i]):  # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with bearish weekly trend and volume spike
            elif (close[i] < lower_d_shifted[i] and 
                  close[i] < ema34_w_aligned[i] and  # Bearish trend: price below weekly EMA34
                  volume[i] > 1.5 * vol_avg_20[i]):  # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to lower Donchian OR weekly trend turns bearish
                if close[i] < lower_d_shifted[i] or close[i] < ema34_w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to upper Donchian OR weekly trend turns bullish
                if close[i] > upper_d_shifted[i] or close[i] > ema34_w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_WeeklyEMA34Trend_Volume"
timeframe = "1d"
leverage = 1.0
#%%