#!/usr/bin/env python3
"""
4h_donchian_breakout_1d_trend_volume_v1
Hypothesis: Daily trend direction + 4H Donchian breakout with volume confirmation.
Go long when daily EMA50 > EMA200 (bullish) and price breaks above 4H Donchian high (20) with volume spike.
Go short when daily EMA50 < EMA200 (bearish) and price breaks below 4H Donchian low (20) with volume spike.
Uses daily trend filter to avoid counter-trend trades and reduce whipsaw in ranging markets.
Target: 25-50 trades/year on 4H timeframe to minimize fee drag.
Works in both bull and bear markets by following the daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Daily EMA50 and EMA200 calculation
    close_1d = df_1d['close'].values
    ema_50d = pd.Series(close_1d).ewm(span=50, min_periods=50).mean().values
    ema_200d = pd.Series(close_1d).ewm(span=200, min_periods=200).mean().values
    
    # Align daily EMAs to 4h timeframe
    ema_50d_aligned = align_htf_to_ltf(prices, df_1d, ema_50d)
    ema_200d_aligned = align_htf_to_ltf(prices, df_1d, ema_200d)
    
    # 4H Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 50-period average
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if data not available
        if (np.isnan(ema_50d_aligned[i]) or np.isnan(ema_200d_aligned[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(volume[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        daily_bullish = ema_50d_aligned[i] > ema_200d_aligned[i]
        daily_bearish = ema_50d_aligned[i] < ema_200d_aligned[i]
        vol_confirmed = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low or daily trend turns bearish
            if close[i] < donch_low[i] or not daily_bullish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high or daily trend turns bullish
            if close[i] > donch_high[i] or not daily_bearish:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with daily bullish trend and volume
            if close[i] > donch_high[i] and daily_bullish and vol_confirmed:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with daily bearish trend and volume
            elif close[i] < donch_low[i] and daily_bearish and vol_confirmed:
                position = -1
                signals[i] = -0.25
    
    return signals