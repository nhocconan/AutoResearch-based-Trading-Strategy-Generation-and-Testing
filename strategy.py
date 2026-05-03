#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian channel breakouts capture strong momentum moves. Trading breakouts
# above the 20-period high or below the 20-period low with 1d EMA50 trend filter
# and volume spike captures sustained moves in both bull and bear markets.
# Designed for 20-50 trades/year on 4h to minimize fee drag while maintaining edge.

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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels on 4h data
    # Highest high and lowest low over past 20 periods (excluding current)
    high_roll = pd.Series(high).rolling(window=20, min_periods=1).max().shift(1).values
    low_roll = pd.Series(low).rolling(window=20, min_periods=1).min().shift(1).values
    
    # Get 4h data for volume confirmation (using current timeframe)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks Donchian high/low with volume spike
        breakout_long = close[i] > high_roll[i] and volume_spike[i]
        breakout_short = close[i] < low_roll[i] and volume_spike[i]
        
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
            if close[i] < low_roll[i] or ema_50_1d_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian high or loses 1d downtrend
            if close[i] > high_roll[i] or ema_50_1d_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals