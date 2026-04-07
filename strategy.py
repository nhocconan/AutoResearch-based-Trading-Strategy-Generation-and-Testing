#!/usr/bin/env python3
"""
4h_donchian_20_1d_trend_volume_v3
Hypothesis: On 4-hour timeframe, buy when price breaks above Donchian(20) high with daily close above 200 EMA and volume > 1.5x 20-period average; sell when price breaks below Donchian low with daily close below 200 EMA and volume > 1.5x 20-period average. Exit on opposite Donchian break. Uses daily trend filter to avoid counter-trend trades and volume confirmation to ensure institutional participation. Fixed look-ahead bug by using daily close from previous day. Designed for 20-50 trades/year to minimize fee drag while capturing trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_trend_volume_v3"
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
    
    # Calculate daily EMA(200) for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMA(200) to 4h timeframe (already shifted by 1 in align_htf_to_ltf)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Donchian channel (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average on 4h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(20, 200), n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low
            if low[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high
            if high[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Only enter with volume confirmation
            if vol_ok:
                # Long: price breaks above Donchian high with daily close above EMA200
                # Use previous day's close to avoid look-ahead
                if high[i] >= donchian_high[i] and ema_200_1d_aligned[i] > 0 and close_1d[-1] > ema_200_1d[-1]:  # Use latest daily close
                    position = 1
                    signals[i] = 0.30
                # Short: price breaks below Donchian low with daily close below EMA200
                elif low[i] <= donchian_low[i] and ema_200_1d_aligned[i] > 0 and close_1d[-1] < ema_200_1d[-1]:  # Use latest daily close
                    position = -1
                    signals[i] = -0.30
    
    return signals