#!/usr/bin/env python3
"""
1d_donchian_20_1w_trend_volume_v1
Hypothesis: On daily timeframe, buy when price breaks above 20-day Donchian upper band with weekly EMA(21) uptrend and volume confirmation; sell/short when price breaks below 20-day Donchian lower band with weekly EMA(21) downtrend and volume confirmation. Uses Donchian breakouts as primary signal, filtered by weekly trend and volume to avoid false breakouts. Designed for 30-100 total trades over 4 years (~7-25/year) to minimize fee drift while capturing trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_20_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 21:
        return np.zeros(n)
    
    # Calculate weekly EMA(21) for trend filter
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align weekly EMA(21) to daily timeframe
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate Donchian channels (20-day)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_21_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band (stop)
            if low[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band (stop)
            if high[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            if vol_ok:
                # Long entry: price breaks above Donchian upper band with weekly uptrend
                if high[i] >= donchian_high[i] and ema_21_1w_aligned[i] > ema_21_1w_aligned[i-1]:
                    position = 1
                    signals[i] = 0.30
                # Short entry: price breaks below Donchian lower band with weekly downtrend
                elif low[i] <= donchian_low[i] and ema_21_1w_aligned[i] < ema_21_1w_aligned[i-1]:
                    position = -1
                    signals[i] = -0.30
    
    return signals