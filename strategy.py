#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above 4h Donchian upper band AND 12h bullish trend (close > EMA50) AND volume > 1.5x 20-period volume EMA
# Short when price breaks below 4h Donchian lower band AND 12h bearish trend (close < EMA50) AND volume > 1.5x 20-period volume EMA
# Uses Donchian channels for clear structure, 12h EMA50 for HTF trend to reduce whipsaw, volume confirmation for momentum
# Target: 20-40 trades/year on 4h (80-160 total over 4 years) to minimize fee drag
# Works in bull markets via longs in bullish 12h trend regime and bear markets via shorts in bearish 12h trend regime

name = "4h_Donchian20_12hTrend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for HTF trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_12h = close_12h > ema_50_12h
    trend_bearish_12h = close_12h < ema_50_12h
    
    # Align 12h trend to 4h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_12h, trend_bullish_12h.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_12h, trend_bearish_12h.astype(float))
    
    # Calculate 4h Donchian channels (20-period)
    # Donchian upper = max(high, 20)
    # Donchian lower = min(low, 20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND 12h bullish trend AND volume spike
            if (close[i] > donchian_upper[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 12h bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND 12h bearish trend AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 12h bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower OR 12h trend turns bearish
            if (close[i] < donchian_lower[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper OR 12h trend turns bullish
            if (close[i] > donchian_upper[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals