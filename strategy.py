#!/usr/bin/env python3
"""
4h_4h_donchian_breakout_12h_volume_v1
Hypothesis: On 4h timeframe, enter long when price breaks above 4h Donchian high (20) with volume > 1.5x 20-period average during 12h uptrend (price above 12h EMA50), enter short when price breaks below 4h Donchian low (20) with volume > 1.5x average during 12h downtrend. Uses volume confirmation and 12h trend filter to avoid false breakouts. Target: 20-40 trades/year to minimize fee drag while capturing strong momentum moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_4h_donchian_breakout_12h_volume_v1"
timeframe = "4h"
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
    
    # Calculate 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend changes
            if close[i] < low_roll[i] or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend changes
            if close[i] > high_roll[i] or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: break above Donchian high in uptrend (price above 12h EMA50)
                if close[i] > high_roll[i] and close[i] > ema_50_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: break below Donchian low in downtrend (price below 12h EMA50)
                elif close[i] < low_roll[i] and close[i] < ema_50_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals