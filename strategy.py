#!/usr/bin/env python3
"""
6h_donchian_20_1d_trend_volume_v1
Hypothesis: On 6-hour timeframe, use Donchian breakouts from 20-period high/low with daily trend filter and volume confirmation.
Go long when price breaks above 20-bar high on 6h chart, daily close > daily EMA(50), and volume > 1.5x 20-period average.
Go short when price breaks below 20-bar low on 6h chart, daily close < daily EMA(50), and volume > 1.5x 20-period average.
Exit when price returns to the 20-bar midpoint on 6h chart.
Designed for 12-37 trades/year to minimize fee decay while capturing trend momentum in both bull and bear markets.
The daily trend filter ensures we trade with higher timeframe momentum, and volume confirmation ensures institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_20_1d_trend_volume_v1"
timeframe = "6h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily EMA(50) to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels on 6h timeframe (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume filter: 20-period average on 6h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to 20-bar midpoint
            if close[i] >= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to 20-bar midpoint
            if close[i] <= donchian_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and daily trend alignment
            if vol_ok:
                # Long: price breaks above 20-bar high with daily uptrend
                if (high[i] > donchian_high[i-1] and close[i] > ema_50_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price breaks below 20-bar low with daily downtrend
                elif (low[i] < donchian_low[i-1] and close[i] < ema_50_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals