#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + Volume + 12h EMA Trend Filter
# Hypothesis: Breakout of 4h Donchian channel with volume confirmation in direction of 12h EMA trend.
# Works in bull/bear by trading with higher timeframe trend. Low trade frequency to avoid fee drag.

name = "4h_donchian_breakout_12h_ema_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA(20) for trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # 4h Donchian(20) channel
    high_4h = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_4h = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 4h volume > 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(ema_20_4h[i]) or np.isnan(high_4h[i]) or np.isnan(low_4h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low or trend changes
            if close[i] < low_4h[i] or close[i] < ema_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high or trend changes
            if close[i] > high_4h[i] or close[i] > ema_20_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout in direction of 12h EMA trend with volume
            if vol_ok:
                if close[i] > ema_20_4h[i]:  # Uptrend
                    if high[i] > high_4h[i] and close[i] > high_4h[i]:  # Breakout above Donchian high
                        position = 1
                        signals[i] = 0.25
                else:  # Downtrend
                    if low[i] < low_4h[i] and close[i] < low_4h[i]:  # Breakdown below Donchian low
                        position = -1
                        signals[i] = -0.25
    
    return signals