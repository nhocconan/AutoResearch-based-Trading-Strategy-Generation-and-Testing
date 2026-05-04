#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when price breaks above 20-day Donchian high AND weekly bullish trend (close > EMA20 weekly) AND volume > 1.5x 20-day volume EMA
# Short when price breaks below 20-day Donchian low AND weekly bearish trend (close < EMA20 weekly) AND volume > 1.5x 20-day volume EMA
# Exit on opposite Donchian breakout or weekly trend reversal
# Uses weekly EMA20 for trend filter to reduce whipsaw, targeting 15-25 trades/year on 1d.
# Volume confirmation (1.5x) reduces noise trades. Donchian channels provide clear breakout structure.
# Works in bull markets via longs in bullish weekly trend regime and bear markets via shorts in bearish weekly trend regime.

name = "1d_Donchian20_1wTrend_VolumeConfirmation"
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
    
    # Get 1d and 1w data for indicators - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate 20-day Donchian channels on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high: max of last 20 days high
    donchian_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian low: min of last 20 days low
    donchian_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (already aligned as we use 1d data)
    # But we need to shift by 1 to avoid look-ahead (use previous day's levels for today's breakout)
    donchian_high_1d = np.roll(donchian_high_1d, 1)
    donchian_low_1d = np.roll(donchian_low_1d, 1)
    donchian_high_1d[0] = np.nan
    donchian_low_1d[0] = np.nan
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_bullish_1w = close_1w > ema_20_1w
    trend_bearish_1w = close_1w < ema_20_1w
    
    # Align weekly trend to 1d timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish_1w.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish_1w.astype(float))
    
    # Calculate volume spike filter (20-day volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup for Donchian calculation
        # Skip if any value is NaN
        if (np.isnan(donchian_high_1d[i]) or np.isnan(donchian_low_1d[i]) or 
            np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND weekly bullish trend AND volume spike
            if (close[i] > donchian_high_1d[i] and 
                trend_bullish_aligned[i] > 0.5 and  # weekly bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND weekly bearish trend AND volume spike
            elif (close[i] < donchian_low_1d[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # weekly bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian low OR weekly trend turns bearish
            if (close[i] < donchian_low_1d[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian high OR weekly trend turns bullish
            if (close[i] > donchian_high_1d[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals