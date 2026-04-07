#!/usr/bin/env python3
"""
4h_donchian_breakout_volume_v1
Hypothesis: On 4h timeframe, enter long when price breaks above 20-period Donchian upper band with volume > 1.5x 20-period average and price > 50-period EMA; enter short when price breaks below 20-period Donchian lower band with volume > 1.5x 20-period average and price < 50-period EMA. Exit when price crosses the 50-period EMA in opposite direction. Uses 12h EMA trend filter to avoid counter-trend trades. Designed for 20-40 trades/year to minimize fee decay while capturing breakouts in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_volume_v1"
timeframe = "4h"
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
    
    # Calculate 20-period Donchian channels
    if len(high) < 20:
        return np.zeros(n)
    
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 50-period EMA for trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA for trend filter (avoid counter-trend trades)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below 50-period EMA
            if close[i] < ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 50-period EMA
            if close[i] > ema_50[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: price breaks above Donchian upper band with price > EMA50 and 12h EMA bullish
                if close[i] > donchian_upper[i] and close[i] > ema_50[i] and ema_50_12h_aligned[i] > ema_50[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below Donchian lower band with price < EMA50 and 12h EMA bearish
                elif close[i] < donchian_lower[i] and close[i] < ema_50[i] and ema_50_12h_aligned[i] < ema_50[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals