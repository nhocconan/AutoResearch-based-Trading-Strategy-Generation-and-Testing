#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian Breakout + 12h EMA Trend + Volume Confirmation
# Hypothesis: Donchian(20) breakouts capture momentum, while 12h EMA filters
# for trend direction and volume confirms institutional participation.
# Works in both bull and bear markets by trading breakouts in the direction
# of the higher timeframe trend. Target: 25-40 trades/year (100-160 total).

name = "4h_donchian_breakout_12h_trend_volume_v1"
timeframe = "4h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian(20) channels on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian or trend turns bearish
            if close[i] < low_20[i] or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian or trend turns bullish
            if close[i] > high_20[i] or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long breakout: price breaks above upper Donchian in uptrend
                if close[i] > high_20[i] and close[i] > ema_50_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price breaks below lower Donchian in downtrend
                elif close[i] < low_20[i] and close[i] < ema_50_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals