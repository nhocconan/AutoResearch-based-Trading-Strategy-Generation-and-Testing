#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with volume spike and 1w EMA50 trend filter
# Uses 1w EMA50 for primary trend direction (avoid counter-trend trades)
# Uses 1d volume > 2.0x 20-period EMA for breakout confirmation
# Designed for 1d timeframe targeting 15-25 trades/year with discrete sizing (0.30)
# Volume spike + trend filter reduces false breakouts while capturing strong moves
# Works in bull markets (breakouts with volume in uptrend) and bear markets (breakouts with volume in downtrend)

name = "1d_Donchian20_VolumeSpike_1wEMA50_Trend"
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
    
    # Get 1d data for Donchian channels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper channel: highest high of last 20 periods
    upper_channel = np.full_like(high_1d, np.nan)
    for i in range(20, len(high_1d)):
        upper_channel[i] = np.max(high_1d[i-20:i])
    
    # Lower channel: lowest low of last 20 periods
    lower_channel = np.full_like(low_1d, np.nan)
    for i in range(20, len(low_1d)):
        lower_channel[i] = np.min(low_1d[i-20:i])
    
    # Align Donchian levels to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # Calculate 1d volume EMA(20) for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_series = pd.Series(vol_1d)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema_20)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    close_series = pd.Series(close_1w)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(vol_ema_20_aligned[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1d volume > 2.0 x 20-period EMA
        volume_confirmed = volume[i] > (2.0 * vol_ema_20_aligned[i])
        
        if position == 0:
            # Long: close breaks above upper Donchian + volume confirmation + price > 1w EMA50 (uptrend)
            if (close[i] > upper_aligned[i] and volume_confirmed and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: close breaks below lower Donchian + volume confirmation + price < 1w EMA50 (downtrend)
            elif (close[i] < lower_aligned[i] and volume_confirmed and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price falls below lower Donchian (mean reversion) OR price < 1w EMA50 (trend change)
            if close[i] < lower_aligned[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price rises above upper Donchian (mean reversion) OR price > 1w EMA50 (trend change)
            if close[i] > upper_aligned[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals