#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian_Breakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period high/low)
    high_1w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_1w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily
    donchian_high_1w = align_htf_to_ltf(prices, df_1w, high_1w)
    donchian_low_1w = align_htf_to_ltf(prices, df_1w, low_1w)
    
    # Weekly EMA50 trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily volume filter: current volume > 1.5 * 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 50)  # Need enough data for Donchian and EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_1w[i]) or np.isnan(donchian_low_1w[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        dh = donchian_high_1w[i]
        dl = donchian_low_1w[i]
        trend = ema50_1w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: break above weekly Donchian high with volume and above weekly trend
            if close[i] > dh and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below weekly Donchian low with volume and below weekly trend
            elif close[i] < dl and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below weekly Donchian low (mean reversion)
            if close[i] < dl:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above weekly Donchian high (mean reversion)
            if close[i] > dh:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals