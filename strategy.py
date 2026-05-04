#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when price breaks above 20-day Donchian upper AND 1w bullish trend (close > EMA50) AND volume > 1.5x 20-day volume EMA
# Short when price breaks below 20-day Donchian lower AND 1w bearish trend (close < EMA50) AND volume > 1.5x 20-day volume EMA
# Uses 1w EMA50 for trend filter to reduce whipsaw, targeting 15-25 trades/year on 1d.
# Volume confirmation and trend filter reduce noise trades. Donchian channels provide robust structure.
# Works in bull markets via longs in bullish 1w trend regime and bear markets via shorts in bearish 1w trend regime.

name = "1d_Donchian20_1wTrend_VolumeSpike"
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
    
    # Get 1w data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_bullish_1w = close_1w > ema_50_1w
    trend_bearish_1w = close_1w < ema_50_1w
    
    # Align 1w trend to 1d timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish_1w.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish_1w.astype(float))
    
    # Calculate 20-day Donchian channels from daily data
    # We need to use daily OHLC for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian upper and lower
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    lookback = 20
    donchian_upper = np.full(len(high_1d), np.nan)
    donchian_lower = np.full(len(low_1d), np.nan)
    
    for i in range(lookback, len(high_1d)):
        donchian_upper[i] = np.max(high_1d[i-lookback:i])
        donchian_lower[i] = np.min(low_1d[i-lookback:i])
    
    # Align Donchian levels to 1d timeframe (no additional delay needed as Donchian uses completed lookback)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Calculate volume spike filter (20-day volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(trend_bullish_aligned[i]) or np.isnan(trend_bearish_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND 1w bullish trend AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1w bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND 1w bearish trend AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1w bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower OR 1w trend turns bearish
            if (close[i] < donchian_lower_aligned[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper OR 1w trend turns bullish
            if (close[i] > donchian_upper_aligned[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals