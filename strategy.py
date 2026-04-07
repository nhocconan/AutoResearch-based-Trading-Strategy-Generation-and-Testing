#!/usr/bin/env python3
"""
4h_donchian_20_1d_trend_volume_v1
Hypothesis: On 4-hour timeframe, use Donchian channel breakout with 1-day trend filter and volume confirmation.
Long when price breaks above Donchian(20) high with 1-day EMA(50) up-trend and volume > 1.5x 20-period average.
Short when price breaks below Donchian(20) low with 1-day EMA(50) down-trend and volume > 1.5x 20-period average.
Exit when price crosses the Donchian midpoint or trend reverses.
Designed for 20-50 trades/year to minimize fee drag while capturing trend continuation.
Works in both bull/bear markets as Donchian adapts to volatility and volume filter ensures institutional participation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_trend_volume_v1"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1-day EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channel (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Align 1d EMA to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume filter: 20-period average on 4h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 50), n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below Donchian midpoint OR trend turns down
            if close[i] < donchian_mid[i] or ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian midpoint OR trend turns up
            if close[i] > donchian_mid[i] or ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: price breaks above Donchian high with up-trend
                if high[i] > donchian_high[i] and ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]:
                    position = 1
                    signals[i] = 0.30
                # Short: price breaks below Donchian low with down-trend
                elif low[i] < donchian_low[i] and ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]:
                    position = -1
                    signals[i] = -0.30
    
    return signals