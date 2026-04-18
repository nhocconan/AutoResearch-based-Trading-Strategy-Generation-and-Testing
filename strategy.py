#!/usr/bin/env python3
"""
12h_Donchian_Breakout_20plus_Volume_Trend_1dEMA34
Hypothesis: On 12h timeframe, break above/below 20-period Donchian channels with volume spike (>1.5x 20-period average) 
and 1d EMA34 trend filter captures sustained directional moves in both bull/bear markets. 
Daily EMA34 filter ensures alignment with higher timeframe trend, reducing whipsaw. 
Target: 15-30 trades/year to minimize fee drag while capturing strong trends with proper risk control.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # 1d EMA34 for trend filter (updated once per day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        ema_1d = np.full(len(df_1d), np.nan)
    else:
        close_1d = pd.Series(df_1d['close'].values)
        ema_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        ema1d = ema_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price breaks above upper channel with volume spike and uptrend (above 1d EMA34)
            if price > upper and vol_spike and price > ema1d:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel with volume spike and downtrend (below 1d EMA34)
            elif price < lower and vol_spike and price < ema1d:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below 1d EMA34 OR touches opposite channel
            if price < ema1d:
                signals[i] = 0.0
                position = 0
            elif price < lower:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above 1d EMA34 OR touches opposite channel
            if price > ema1d:
                signals[i] = 0.0
                position = 0
            elif price > upper:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_Breakout_20plus_Volume_Trend_1dEMA34"
timeframe = "12h"
leverage = 1.0