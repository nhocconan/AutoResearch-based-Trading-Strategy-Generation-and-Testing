#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchianBreakout_Trend_Volume_v2"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period EMA for weekly trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels from previous daily bar
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    
    # Upper and lower bands from previous 20 days (excluding current)
    donchian_high = high_1d.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = low_1d.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Daily volume average for confirmation
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Break above Donchian high with volume and above weekly EMA trend
            if close[i] > donchian_high[i] and vol_ok and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with volume and below weekly EMA trend
            elif close[i] < donchian_low[i] and vol_ok and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below Donchian low or trend reversal
            if close[i] < donchian_low[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above Donchian high or trend reversal
            if close[i] > donchian_high[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals