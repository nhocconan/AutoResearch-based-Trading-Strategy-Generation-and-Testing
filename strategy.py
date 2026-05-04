#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when price breaks above 20-day Donchian high AND weekly bullish trend (close > weekly EMA50) AND volume > 1.5x 20-day volume EMA
# Short when price breaks below 20-day Donchian low AND weekly bearish trend (close < weekly EMA50) AND volume > 1.5x 20-day volume EMA
# Uses weekly EMA50 for trend filter to reduce whipsaw, targeting 10-25 trades/year on 1d.
# Volume confirmation (1.5x) reduces noise trades. Donchian channels provide robust structure.
# Works in bull markets via longs in bullish weekly trend regime and bear markets via shorts in bearish weekly trend regime.

name = "1d_Donchian20_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1w = close_1w > ema_50_1w
    trend_bearish_1w = close_1w < ema_50_1w
    
    # Align weekly trend to 1d timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish_1w.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish_1w.astype(float))
    
    # Calculate 20-day Donchian channels from daily data
    # Need to use rolling window on daily prices
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike filter (20-day volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(donchian_high_20[i]) or np.isnan(donchian_low_20[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 20-day Donchian high AND weekly bullish trend AND volume spike
            if (close[i] > donchian_high_20[i] and 
                trend_bullish_aligned[i] > 0.5 and  # weekly bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below 20-day Donchian low AND weekly bearish trend AND volume spike
            elif (close[i] < donchian_low_20[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # weekly bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below 20-day Donchian low OR weekly trend turns bearish
            if (close[i] < donchian_low_20[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above 20-day Donchian high OR weekly trend turns bullish
            if (close[i] > donchian_high_20[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals