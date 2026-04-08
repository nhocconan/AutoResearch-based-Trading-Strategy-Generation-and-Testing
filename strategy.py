#!/usr/bin/env python3
# 4h_12h_donchian_breakout_volume_trend_v2
# Hypothesis: Price breaking Donchian channel (20) on 4h with volume > 1.5x 20-period average and 12h trend filter (close > 12h SMA(50) for long, close < 12h SMA(50) for short).
# Works in bull markets via upside breakouts and bear markets via downside breakdowns.
# Volume confirmation filters false breakouts; 12h trend reduces whipsaw. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_donchian_breakout_volume_trend_v2"
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
    
    # Donchian channel (20) on 4h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume MA(20) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h SMA(50) for trend filter
    sma50_12h = pd.Series(close_12h).rolling(window=50, min_periods=50).mean().values
    sma50_12h_aligned = align_htf_to_ltf(prices, df_12h, sma50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i]) or np.isnan(sma50_12h_aligned[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or trend reverses
            if close[i] < donchian_low[i] or close[i] < sma50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or trend reverses
            if close[i] > donchian_high[i] or close[i] > sma50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with volume surge and uptrend
            if close[i] > donchian_high[i] and vol_surge and close[i] > sma50_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume surge and downtrend
            elif close[i] < donchian_low[i] and vol_surge and close[i] < sma50_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals