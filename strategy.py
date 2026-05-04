#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when price breaks above 20-day Donchian high AND weekly bullish trend (close > weekly EMA34) AND volume > 1.5x 20-day volume EMA
# Short when price breaks below 20-day Donchian low AND weekly bearish trend (close < weekly EMA34) AND volume > 1.5x 20-day volume EMA
# Exit on opposite Donchian breakout or trend reversal
# Uses weekly EMA34 for trend filter to reduce whipsaw in choppy markets, targeting 15-25 trades/year on 1d.
# Volume confirmation (1.5x) reduces false breakouts. Donchian channels provide clear structure in both bull/bear regimes.
# Works in bull markets via longs in bullish weekly trend and bear markets via shorts in bearish weekly trend.

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
    
    # Get 1d data for Donchian calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 20-period Donchian channels on daily data
    # Donchian High = max(high over last 20 days)
    # Donchian Low = min(low over last 20 days)
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume EMA for confirmation
    vol_ema_20 = pd.Series(volume_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume_1d > (vol_ema_20 * 1.5)  # Volume at least 1.5x average for confirmation
    
    # Get 1w data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_bullish_1w = close_1w > ema_34_1w
    trend_bearish_1w = close_1w < ema_34_1w
    
    # Align 1d indicators to 1d timeframe (no alignment needed as we're already on 1d)
    # But we need to align weekly trend to daily timeframe
    trend_bullish_aligned = align_htf_to_ltf(prices, df_1w, trend_bullish_1w.astype(float))
    trend_bearish_aligned = align_htf_to_ltf(prices, df_1w, trend_bearish_1w.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(trend_bullish_aligned[i]) or 
            np.isnan(trend_bearish_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high AND weekly bullish trend AND volume spike
            if (close[i] > donchian_high[i] and 
                trend_bullish_aligned[i] > 0.5 and  # Weekly bullish trend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low AND weekly bearish trend AND volume spike
            elif (close[i] < donchian_low[i] and 
                  trend_bearish_aligned[i] > 0.5 and  # Weekly bearish trend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low OR weekly trend turns bearish
            if (close[i] < donchian_low[i] or 
                trend_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high OR weekly trend turns bullish
            if (close[i] > donchian_high[i] or 
                trend_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals