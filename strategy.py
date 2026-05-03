#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter (EMA50) and volume confirmation
# Donchian breakouts capture strong momentum moves. Using 1d EMA50 as trend filter ensures
# we only trade in the direction of the higher timeframe trend. Volume confirmation
# filters out false breakouts. Designed for 20-50 trades/year on 4h to minimize fee drag.
# Works in both bull and bear markets by following the 1d trend.

name = "4h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    # Upper band = max(high, lookback=20)
    # Lower band = min(low, lookback=20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume spike above 2.0 x 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks Donchian bands with volume spike
        breakout_long = close[i] > donchian_upper[i] and volume_spike[i]
        breakout_short = close[i] < donchian_lower[i] and volume_spike[i]
        
        if position == 0:
            # Long: break above upper band in 1d uptrend with volume spike
            if breakout_long and ema_50_1d_aligned[i] > close[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower band in 1d downtrend with volume spike
            elif breakout_short and ema_50_1d_aligned[i] < close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower band or loses 1d uptrend
            if close[i] < donchian_lower[i] or ema_50_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above upper band or loses 1d downtrend
            if close[i] > donchian_upper[i] or ema_50_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals