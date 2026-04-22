#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation
# Donchian breakout captures strong directional moves.
# 1w EMA(50) ensures alignment with weekly trend to avoid counter-trend trades.
# Volume > 1.5x average confirms breakout strength.
# Works in both bull and bear markets by following the weekly trend.
# Uses discrete position sizing (0.25) to minimize fee churn.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Donchian calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d Donchian(20) channels
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high + above weekly EMA + volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + below weekly EMA + volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses weekly EMA or volatility drop
            if position == 1:
                # Exit long: Price below weekly EMA or low volume
                if close[i] < ema_50_1w_aligned[i] or volume[i] < vol_avg_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Price above weekly EMA or low volume
                if close[i] > ema_50_1w_aligned[i] or volume[i] < vol_avg_20[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0