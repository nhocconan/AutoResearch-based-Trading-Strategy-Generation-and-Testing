#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# Uses Donchian channels for breakout signals, filtered by 1d EMA50 trend and 6h volume spikes.
# Designed to capture strong momentum moves in both bull and bear markets while avoiding chop.
# Target: 12-30 trades/year per symbol (50-120 total over 4 years) to minimize fee drag.

name = "6h_Donchian20_1dEMA50_VolumeSpike_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 6h data for Donchian channels (20-period)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate 6h Donchian channels
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Upper channel: highest high over past 20 periods
    upper_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low over past 20 periods
    lower_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_6h, upper_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_6h, lower_20)
    
    # Get 6h data for volume EMA(20) for volume confirmation
    vol_6h = df_6h['volume'].values
    vol_ema_20 = pd.Series(vol_6h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20_aligned[i])
        
        # 1d trend: bullish if close > EMA50, bearish if close < EMA50
        bullish_trend = close[i] > ema_50_1d_aligned[i]
        bearish_trend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + bullish 1d trend
            if (close[i] > upper_20_aligned[i] and volume_confirmed and bullish_trend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation + bearish 1d trend
            elif (close[i] < lower_20_aligned[i] and volume_confirmed and bearish_trend):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls below lower Donchian OR 1d trend turns bearish
            if close[i] < lower_20_aligned[i] or bearish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises above upper Donchian OR 1d trend turns bullish
            if close[i] > upper_20_aligned[i] or bullish_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals