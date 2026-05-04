#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when price breaks above 20-day Donchian upper band AND weekly bullish trend (close > weekly EMA34) AND volume > 1.5x 20-day volume EMA
# Short when price breaks below 20-day Donchian lower band AND weekly bearish trend (close < weekly EMA34) AND volume > 1.5x 20-day volume EMA
# Uses weekly EMA34 for trend filter to reduce whipsaw, targeting 10-25 trades/year on 1d.
# Volume confirmation (1.5x) reduces noise trades. Donchian channels provide clear structure.
# Works in bull markets via longs in bullish weekly trend regime and bear markets via shorts in bearish weekly trend regime.

name = "1d_Donchian20_1wTrend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian calculation and volume EMA - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on 1d
    donchian_upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume EMA on 1d
    vol_ema_20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume_1d > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    # Align 1d indicators to 1d timeframe (already aligned, but for consistency)
    donchian_upper_aligned = donchian_upper
    donchian_lower_aligned = donchian_lower
    volume_spike_aligned = volume_spike
    
    # Get 1w data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_bullish_1w = close_1w > ema_34_1w
    trend_bearish_1w = close_1w < ema_34_1w
    
    # Align 1w trend to 1d timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish_1w.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(trend_bullish_aligned[i]) or 
            np.isnan(trend_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND weekly bullish trend AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                trend_bullish_aligned[i] > 0.5 and  # Weekly bullish trend
                volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND weekly bearish trend AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # Weekly bearish trend
                  volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower OR weekly trend turns bearish
            if (close[i] < donchian_lower_aligned[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper OR weekly trend turns bullish
            if (close[i] > donchian_upper_aligned[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals