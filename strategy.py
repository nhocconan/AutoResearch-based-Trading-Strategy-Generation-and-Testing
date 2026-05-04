#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend + volume confirmation
# In uptrend (price > 1d EMA34), go long on upper Donchian breakout with volume > 1.5x 20 EMA.
# In downtrend (price < 1d EMA34), go short on lower Donchian breakout with volume > 1.5x 20 EMA.
# Exit on opposite Donchian breakout or trend reversal. Designed for 4h timeframe targeting 75-200 trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "4h_Donchian20_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Determine trend: price above/below 1d EMA34
            if close[i] > ema_34_aligned[i]:
                # Uptrend: look for long on upper Donchian breakout
                if close[i] > highest_20[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
            else:
                # Downtrend: look for short on lower Donchian breakout
                if close[i] < lowest_20[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian OR trend reverses to downtrend
            if close[i] < lowest_20[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian OR trend reverses to uptrend
            if close[i] > highest_20[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals