#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Donchian channels provide robust breakout levels based on price extremes
# 1w EMA50 ensures we only trade in the direction of the higher timeframe trend
# Volume confirmation requires 2.0x average volume to ensure strong participation
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag on 1d timeframe
# Works in both bull and bear markets by following the 1w trend direction and using breakout/mean reversion logic

name = "1d_Donchian20_1wEMA50_VolumeConfirm"
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
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels from previous 1d bar (20-period)
    # Upper = max(high over last 20 periods)
    # Lower = min(low over last 20 periods)
    # Using rolling window with min_periods to avoid look-ahead
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: 50-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_50 = vol_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 50-period EMA (strict to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_50[i])
        
        # Donchian breakout signals with 1w trend filter
        # Long: price breaks above upper Donchian + volume spike + price above 1w EMA50
        # Short: price breaks below lower Donchian + volume spike + price below 1w EMA50
        if position == 0:
            if (close[i] > donchian_upper[i] and volume_spike and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            elif (close[i] < donchian_lower[i] and volume_spike and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower Donchian (mean reversion) OR price below 1w EMA50
            if close[i] < donchian_lower[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper Donchian (mean reversion) OR price above 1w EMA50
            if close[i] > donchian_upper[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals