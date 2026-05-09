#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian_Breakout_Trend_Volume"
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
    
    # Get weekly data for Donchian and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly trend (50 EMA)
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume filter: current volume > 1.5 * 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Align weekly indicators to daily
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 50)  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper = high_20_aligned[i]
        lower = low_20_aligned[i]
        trend = ema50_1w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: break above upper band with volume and above trend
            if close[i] > upper and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: break below lower band with volume and below trend
            elif close[i] < lower and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below lower band (mean reversion)
            if close[i] < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above upper band (mean reversion)
            if close[i] > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals