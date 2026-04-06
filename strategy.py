#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above 12h Donchian upper band AND 1d EMA(50) > EMA(200) AND volume > 1.5x average
# Short when price breaks below 12h Donchian lower band AND 1d EMA(50) < EMA(200) AND volume > 1.5x average
# Exit when price returns to Donchian middle or trend reverses
# Uses trend filter to avoid whipsaws in ranging markets, targeting 75-200 total trades over 4 years
# Works in bull markets via trend-following breaks and in bear via short breaks

name = "12h_donchian_ema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    middle = (highest_high + lowest_low) / 2
    
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_middle = middle.values
    
    # 1d EMA trend filter (50 and 200)
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    ema_50 = pd.Series(daily_close).ewm(span=50, adjust=False).mean().values
    ema_200 = pd.Series(daily_close).ewm(span=200, adjust=False).mean().values
    
    # Align daily EMA to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if (close[i] <= donchian_middle[i] or ema_50_aligned[i] < ema_200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if (close[i] >= donchian_middle[i] or ema_50_aligned[i] > ema_200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: price breaks above Donchian upper + uptrend + volume
            if (close[i] > donchian_upper[i] and 
                ema_50_aligned[i] > ema_200_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower + downtrend + volume
            elif (close[i] < donchian_lower[i] and 
                  ema_50_aligned[i] < ema_200_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals