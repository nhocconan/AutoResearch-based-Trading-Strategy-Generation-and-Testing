#!/usr/bin/env python3
"""
12h_donchian_20_1d_trend_volume_v1
Hypothesis: On 12-hour timeframe, use 20-period Donchian channel breakouts for trend continuation, filtered by 1-day EMA trend and volume confirmation. This strategy captures strong momentum moves in both bull and bear markets, with volume ensuring institutional participation. Designed for 50-150 total trades over 4 years (~12-37/year) to minimize fee drag while performing in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_20_1d_trend_volume_v1"
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA(50) to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channel (20-period) on 12h timeframe
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 24-period average on 12h timeframe (equivalent to ~12 days)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(24, 50), n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low (stop loss)
            if low[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high (stop loss)
            if high[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long breakout: price breaks above Donchian high with upward EMA bias
                if high[i] >= donchian_high[i] and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price breaks below Donchian low with downward EMA bias
                elif low[i] <= donchian_low[i] and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals