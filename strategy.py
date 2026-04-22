#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with 1-day volume confirmation and trend filter.
Long when price breaks above 20-period Donchian high + 1-day volume > average + 1-day close > 50-period EMA.
Short when price breaks below 20-period Donchian low + 1-day volume > average + 1-day close < 50-period EMA.
Exit when price crosses opposite Donchian level or volume drops below average.
Combines price channel breakout with institutional volume confirmation and trend filter for robustness.
Works in bull markets via breakouts and bear markets via breakdowns with volume validation.
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
    
    # Load 1-day data for volume and trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-day volume average (50-period)
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=50, min_periods=50).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # 1-day close and 50-period EMA for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12-hour Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume confirmation + uptrend
            if (close[i] > donchian_high[i] and 
                volume_1d[i] > avg_vol_1d_aligned[i] and 
                close_1d[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume confirmation + downtrend
            elif (close[i] < donchian_low[i] and 
                  volume_1d[i] > avg_vol_1d_aligned[i] and 
                  close_1d[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below Donchian low OR volume drops below average
                if (close[i] < donchian_low[i] or 
                    volume_1d[i] <= avg_vol_1d_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above Donchian high OR volume drops below average
                if (close[i] > donchian_high[i] or 
                    volume_1d[i] <= avg_vol_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_Breakout_1dVolume_EMA_Trend"
timeframe = "12h"
leverage = 1.0