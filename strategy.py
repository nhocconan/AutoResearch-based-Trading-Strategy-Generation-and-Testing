#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Donchian breakout with 4h trend filter and volume confirmation
# Long when price breaks above 20-period Donchian high AND 4h bullish trend (close > EMA50) AND volume > 1.5x 20-period volume SMA
# Short when price breaks below 20-period Donchian low AND 4h bearish trend (close < EMA50) AND volume > 1.5x 20-period volume SMA
# Uses Donchian channels for clear breakout structure, 4h EMA50 for trend filter to avoid counter-trend whipsaw
# Volume confirmation ensures breakouts have conviction. Target 15-35 trades/year on 1h timeframe.
# Works in bull markets via longs in bullish 4h trend and bear markets via shorts in bearish 4h trend.

name = "1h_Donchian20_4hTrend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 60:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_4h = close_4h > ema_50_4h
    trend_bearish_4h = close_4h < ema_50_4h
    
    # Align 4h trend to 1h timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_4h, trend_bullish_4h.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_4h, trend_bearish_4h.astype(float))
    
    # Calculate Donchian channels (20-period) on 1h timeframe
    # Donchian High = highest high over past 20 periods
    # Donchian Low = lowest low over past 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation (20-period volume SMA)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (vol_sma_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND 4h bullish trend AND volume confirmation
            if (close[i] > donchian_high[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 4h bullish trend
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Donchian low AND 4h bearish trend AND volume confirmation
            elif (close[i] < donchian_low[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 4h bearish trend
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian low OR 4h trend turns bearish
            if (close[i] < donchian_low[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price closes above Donchian high OR 4h trend turns bullish
            if (close[i] > donchian_high[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals