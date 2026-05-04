#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when price breaks above 20-day Donchian high AND 1w bullish trend (close > EMA50) AND volume > 1.5x 20-day volume EMA
# Short when price breaks below 20-day Donchian low AND 1w bearish trend (close < EMA50) AND volume > 1.5x 20-day volume EMA
# Exit on opposite Donchian breakout or trend reversal
# Uses discrete position sizing (0.25) to control risk and minimize fee churn
# Target: 15-25 trades/year on 1d timeframe to avoid fee drag while capturing major trends
# Works in bull markets via longs in bullish 1w trend and bear markets via shorts in bearish 1w trend

name = "1d_Donchian20_1wTrend_VolumeConfirm"
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
    
    # Get 1d data for Donchian channels and volume EMA - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on 1d
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume EMA on 1d
    vol_ema_20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume_1d > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    # Align 1d indicators to 1d timeframe (same timeframe, but need to wait for bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or np.isnan(trend_bullish_aligned[i]) or 
            np.isnan(trend_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND 1w bullish trend AND volume spike
            if (close[i] > donchian_high_aligned[i] and 
                trend_bullish_aligned[i] > 0.5 and  # 1w bullish trend
                volume_spike_aligned[i] > 0.5):     # volume confirmation
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND 1w bearish trend AND volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # 1w bearish trend
                  volume_spike_aligned[i] > 0.5):     # volume confirmation
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR 1w trend turns bearish
            if (close[i] < donchian_low_aligned[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR 1w trend turns bullish
            if (close[i] > donchian_high_aligned[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals