#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume spike confirmation and 12h EMA50 trend filter
# Uses 12h EMA > EMA50 for trend direction (avoid counter-trend trades)
# Uses 4h volume > 2.0x 20-period EMA for strong confirmation
# Designed for 4h timeframe targeting 25-35 trades/year with discrete sizing (0.30)
# Volume spike + 12h trend filter reduces false breakouts while maintaining alignment with higher timeframe trend
# Works in bull markets (breakouts with volume in uptrend) and bear markets (breakouts with volume in downtrend)

name = "4h_Donchian20_VolumeSpike_12hEMA50_Trend"
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
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = get_htf_data(prices, '4h')['high'].values
    low_4h = get_htf_data(prices, '4h')['low'].values
    
    # Upper channel: highest high over past 20 periods
    upper_channel = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low over past 20 periods
    lower_channel = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    upper_channel_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), lower_channel)
    
    # Calculate 4h volume EMA(20) for volume confirmation
    df_4h = get_htf_data(prices, '4h')
    vol_4h = df_4h['volume'].values
    vol_ema_20 = pd.Series(vol_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_4h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20_aligned[i])
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + 12h EMA > EMA50 (uptrend)
            if (close[i] > upper_channel_aligned[i] and volume_confirmed and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation + 12h EMA < EMA50 (downtrend)
            elif (close[i] < lower_channel_aligned[i] and volume_confirmed and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price falls below lower Donchian (mean reversion) OR 12h EMA < EMA50 (trend change)
            if close[i] < lower_channel_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above upper Donchian (mean reversion) OR 12h EMA > EMA50 (trend change)
            if close[i] > upper_channel_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals