#!/usr/bin/env python3
# 4h_1d_donchian_breakout_volume_trend_v1
# Hypothesis: Price breaking Donchian channel (20) on 4h with volume confirmation and 1-day trend filter.
# Long when price breaks above Donchian upper band with volume > 1.5x 20-period average and 1d close > 1d SMA(50).
# Short when price breaks below Donchian lower band with volume > 1.5x 20-period average and 1d close < 1d SMA(50).
# Uses 4h timeframe for entries and 1d for trend filter to reduce whipsaw. Designed for 20-50 trades/year.
# Works in bull markets via upside breakouts and bear markets via downside breakdowns.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_donchian_breakout_volume_trend_v1"
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
    
    # 4h Donchian channel (20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume MA(20) for volume confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    # Calculate 1d SMA(50) for trend filter
    sma50_1d = pd.Series(df_1d['close'].values).rolling(window=50, min_periods=50).mean().values
    # Align 1d SMA(50) to 4h timeframe
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure SMA(50) and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i]) or np.isnan(sma50_1d_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian lower band or trend reverses
            if close[i] < donchian_low[i] or close[i] < sma50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian upper band or trend reverses
            if close[i] > donchian_high[i] or close[i] > sma50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian upper band with volume surge and uptrend
            if close[i] > donchian_high[i] and vol_surge and close[i] > sma50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian lower band with volume surge and downtrend
            elif close[i] < donchian_low[i] and vol_surge and close[i] < sma50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals