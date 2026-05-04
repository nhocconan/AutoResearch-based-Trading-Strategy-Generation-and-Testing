#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with volume spike confirmation and 1w EMA50 trend filter
# Uses 1w EMA > EMA50 for trend direction to avoid counter-trend trades
# Uses 12h volume > 2.0x 20-period EMA for strong confirmation
# Designed for 12h timeframe targeting 12-37 trades/year with discrete sizing (0.30)
# Volume spike + 1w trend filter reduces false breakouts while maintaining alignment with higher timeframe trend
# Works in bull markets (breakouts with volume in uptrend) and bear markets (breakouts with volume in downtrend)

name = "12h_Donchian20_VolumeSpike_1wEMA50_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Get 12h data for Donchian channels and volume EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper channel: highest high over past 20 periods
    upper_channel = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low over past 20 periods
    lower_channel = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    upper_channel_aligned = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_12h, lower_channel)
    
    # Calculate 12h volume EMA(20) for volume confirmation
    vol_12h = df_12h['volume'].values
    vol_ema_20 = pd.Series(vol_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_12h, vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or np.isnan(vol_ema_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20_aligned[i])
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume confirmation + 1w EMA > EMA50 (uptrend)
            if (close[i] > upper_channel_aligned[i] and volume_confirmed and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: price breaks below lower Donchian + volume confirmation + 1w EMA < EMA50 (downtrend)
            elif (close[i] < lower_channel_aligned[i] and volume_confirmed and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price falls below lower Donchian (mean reversion) OR 1w EMA < EMA50 (trend change)
            if close[i] < lower_channel_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above upper Donchian (mean reversion) OR 1w EMA > EMA50 (trend change)
            if close[i] > upper_channel_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals