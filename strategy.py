#!/usr/bin/env python3
"""
12h_donchian_breakout_1w_trend_volume_v1
Hypothesis: Donchian(20) breakout on 12h captures trend continuation. Long when price breaks above 20-period high and above 1w EMA100 (uptrend). Short when price breaks below 20-period low and below 1w EMA100 (downtrend). Volume confirmation filters weak signals. Works in bull/bear by following higher timeframe trend. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA100 for trend filter
    ema_100 = df_1w['close'].ewm(span=100, adjust=False).mean()
    
    # Align 1w EMA100 to 12h timeframe
    ema_100_aligned = align_htf_to_ltf(prices, df_1w, ema_100.values)
    
    # Donchian(20) on 12h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # Volume confirmation (20-period average on 12h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_100_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or below EMA100
            if close[i] < donchian_low[i] or close[i] < ema_100_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or above EMA100
            if close[i] > donchian_high[i] or close[i] > ema_100_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high, with volume and above EMA100
            if (close[i] > donchian_high[i] and vol_confirm and 
                close[i] > ema_100_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low, with volume and below EMA100
            elif (close[i] < donchian_low[i] and vol_confirm and 
                  close[i] < ema_100_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals