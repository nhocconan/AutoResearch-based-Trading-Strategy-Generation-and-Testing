#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w trend filter and volume confirmation
# Donchian channels provide clear structural breakout levels. Trading breakouts
# of the 20-day high/low with 1-week EMA50 trend filter and volume confirmation
# captures strong momentum moves while minimizing whipsaws. Designed for 15-25 trades/year
# on 1d timeframe to reduce fee drag and improve test generalization.

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian(20) channels
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Get 1d data for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_rolling_max[i]) or 
            np.isnan(low_rolling_min[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks Donchian channels with volume spike
        breakout_long = close[i] > high_rolling_max[i] and volume_spike[i]
        breakout_short = close[i] < low_rolling_min[i] and volume_spike[i]
        
        if position == 0:
            # Long: break above Donchian high in 1w uptrend with volume spike
            if breakout_long and ema_50_1w_aligned[i] > close[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low in 1w downtrend with volume spike
            elif breakout_short and ema_50_1w_aligned[i] < close[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian low or loses 1w uptrend
            if close[i] < low_rolling_min[i] or ema_50_1w_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian high or loses 1w downtrend
            if close[i] > high_rolling_max[i] or ema_50_1w_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals