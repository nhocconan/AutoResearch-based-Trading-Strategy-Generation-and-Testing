#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly EMA21 for trend
    ema21_weekly = pd.Series(df_weekly['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Weekly Donchian channels (20 period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Weekly volume average for volume filter
    vol_weekly = df_weekly['volume'].values
    vol_avg_weekly = pd.Series(vol_weekly).rolling(window=20, min_periods=20).mean().values
    
    # Align all to daily
    ema21_weekly_daily = align_htf_to_ltf(prices, df_weekly, ema21_weekly)
    donchian_high_daily = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_daily = align_htf_to_ltf(prices, df_weekly, donchian_low)
    vol_avg_weekly_daily = align_htf_to_ltf(prices, df_weekly, vol_avg_weekly)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(ema21_weekly_daily[i]) or np.isnan(donchian_high_daily[i]) or 
            np.isnan(donchian_low_daily[i]) or np.isnan(vol_avg_weekly_daily[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema21_weekly_daily[i]
        upper_band = donchian_high_daily[i]
        lower_band = donchian_low_daily[i]
        vol_avg = vol_avg_weekly_daily[i]
        vol_ok = volume[i] > vol_avg * 1.5
        
        if position == 0:
            # Long: break above weekly Donchian high with volume and above weekly EMA21
            if close[i] > upper_band and vol_ok and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Short: break below weekly Donchian low with volume and below weekly EMA21
            elif close[i] < lower_band and vol_ok and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below weekly Donchian low or trend reversal
            if close[i] < lower_band or close[i] < trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above weekly Donchian high or trend reversal
            if close[i] > upper_band or close[i] > trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals