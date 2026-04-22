#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA to daily timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above 20-day high with volume AND above weekly EMA50 (uptrend)
            if (close[i] > high_20[i] and volume[i] > 1.5 * vol_avg_20[i] and close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low with volume AND below weekly EMA50 (downtrend)
            elif (close[i] < low_20[i] and volume[i] > 1.5 * vol_avg_20[i] and close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite Donchian level
            if position == 1:
                # Exit long: Price closes below 20-day low
                if close[i] < low_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price closes above 20-day high
                if close[i] > high_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0