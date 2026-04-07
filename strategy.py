#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# Uses 12h price channel breakouts (Donchian 20) in the direction of 1d EMA trend
# Volume confirmation ensures breakouts are genuine
# Designed for low frequency (target: 15-35 trades/year) to minimize fee impact
# Works in bull markets via breakout continuation and in bear via trend-following shorts

name = "12h_donchian20_1d_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h Donchian channels (20-period)
    high_12h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_12h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(high_12h[i]) or np.isnan(low_12h[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Trend filter from 1d EMA
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Exit conditions
        if position == 1:  # Long position
            # Exit on trend reversal or at opposite Donchian level
            if not uptrend or close[i] < low_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit on trend reversal or at opposite Donchian level
            if not downtrend or close[i] > high_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Breakout entry in direction of trend
            if uptrend and vol_confirm and close[i] > high_12h[i]:
                position = 1
                signals[i] = 0.25
            elif downtrend and vol_confirm and close[i] < low_12h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals