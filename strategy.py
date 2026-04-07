#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v2
Hypothesis: On 4h timeframe, buy when price breaks above 20-period Donchian high with 1d uptrend and volume confirmation; sell when price breaks below 20-period Donchian low with 1d downtrend and volume confirmation. Uses Donchian breakouts for trend following, filtered by daily EMA trend and volume to avoid false signals. Works in bull markets via breakout continuation and in bear markets via shorting breakdowns. Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity with fee minimization.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v2"
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
    
    # Calculate EMA on daily timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate Donchian channels (20-period) on 4h timeframe
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        # Trend filter: price above/below daily EMA
        above_ema = close[i] > ema_1d_aligned[i]
        below_ema = close[i] < ema_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low (trend reversal)
            if close[i] < low_min[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high (trend reversal)
            if close[i] > high_max[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout above Donchian high with uptrend - go long
                if close[i] > high_max[i] and above_ema:
                    position = 1
                    signals[i] = 0.25
                # Breakdown below Donchian low with downtrend - go short
                elif close[i] < low_min[i] and below_ema:
                    position = -1
                    signals[i] = -0.25
    
    return signals