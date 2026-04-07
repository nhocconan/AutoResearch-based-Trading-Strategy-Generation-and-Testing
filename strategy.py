#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_volume_v1
Hypothesis: On 1d timeframe, break above/below Donchian(20) channels with 1w trend filter and volume confirmation captures sustained moves while avoiding whipsaws. Daily timeframe reduces noise and transaction costs, with trend filter ensuring alignment with weekly momentum. Target: 20-80 total trades over 4 years (5-20/year) to minimize fee drag while performing in bull, bear, and sideways markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
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
    
    # Donchian channels (20-period)
    lookback = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=lookback, min_periods=lookback).max().values
    lower = low_series.rolling(window=lookback, min_periods=lookback).min().values
    
    # 1-week trend filter: EMA(21) on weekly data
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume confirmation: 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(lookback, 21, 20), n):
        # Skip if data not available
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian or weekly trend turns bearish
            if close[i] < lower[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian or weekly trend turns bullish
            if close[i] > upper[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish breakout: price closes above upper Donchian with weekly uptrend
                if close[i] > upper[i] and close[i] > ema_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.30
                # Bearish breakout: price closes below lower Donchian with weekly downtrend
                elif close[i] < lower[i] and close[i] < ema_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.30
    
    return signals