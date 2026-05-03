#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# Donchian channels provide robust structure for breakout trading. Trading breakouts
# above the 20-period high or below the 20-period low with 1d EMA50 trend filter and
# volume spike captures strong moves while minimizing whipsaw. Designed for 12-37 trades/year
# on 12h timeframe to stay within fee drag limits and work in both bull and bear markets.

name = "12h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Get 12h data for Donchian channel (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian levels: 20-period high and low
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Rolling max/min for Donchian channel
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get volume confirmation (20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks Donchian high/low with volume spike
        breakout_long = close[i] > donchian_high_aligned[i] and volume_spike[i]
        breakout_short = close[i] < donchian_low_aligned[i] and volume_spike[i]
        
        if position == 0:
            # Long: break above Donchian high in 1d uptrend with volume spike
            if breakout_long and ema_50_1d_aligned[i] > close[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low in 1d downtrend with volume spike
            elif breakout_short and ema_50_1d_aligned[i] < close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian low or loses 1d uptrend
            if close[i] < donchian_low_aligned[i] or ema_50_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian high or loses 1d downtrend
            if close[i] > donchian_high_aligned[i] or ema_50_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals