#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian20_Breakout_WeekTrend_Volume"
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
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # Weekly high and low for Donchian channel (20-period)
    weekly_high = df_w['high'].values
    weekly_low = df_w['low'].values
    
    # Calculate Donchian upper and lower bands (20-period)
    high_series = pd.Series(weekly_high)
    low_series = pd.Series(weekly_low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA(50) for trend filter
    weekly_close = df_w['close'].values
    close_w = pd.Series(weekly_close)
    ema50_w = close_w.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Align to daily timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_w, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_w, donchian_lower)
    ema50_w_aligned = align_htf_to_ltf(prices, df_w, ema50_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema50_w_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume and above weekly EMA trend
            if close[i] > donchian_upper_aligned[i] and vol_ok and close[i] > ema50_w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with volume and below weekly EMA trend
            elif close[i] < donchian_lower_aligned[i] and vol_ok and close[i] < ema50_w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price crosses below lower Donchian (reversion to mean)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price crosses above upper Donchian (reversion to mean)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals