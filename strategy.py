#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_donchian_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period Donchian channels on weekly data
    high_20w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe (shifted by 1 to avoid look-ahead)
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Volume filter - 20-period average on 12h data
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    # Trend filter: 50-period SMA on 12h data
    close_series = pd.Series(close)
    sma_50 = close_series.rolling(window=50, min_periods=50).mean().values
    trend_up = close > sma_50
    trend_down = close < sma_50
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if not ready
        if (np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or
            np.isnan(volume_ok[i]) or np.isnan(trend_up[i]) or np.isnan(trend_down[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price breaks above upper Donchian with volume confirmation and uptrend
        long_signal = close[i] > high_20w_aligned[i] and volume_ok[i] and trend_up[i]
        # Short: price breaks below lower Donchian with volume confirmation and downtrend
        short_signal = close[i] < low_20w_aligned[i] and volume_ok[i] and trend_down[i]
        
        # Exit when price returns to middle of channel (mean reversion)
        mid_20w = (high_20w_aligned[i] + low_20w_aligned[i]) / 2.0
        exit_long = close[i] < mid_20w
        exit_short = close[i] > mid_20w
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals