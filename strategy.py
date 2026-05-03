#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Donchian breakouts capture strong directional moves. 1w EMA34 filter ensures alignment with weekly trend.
# Volume confirmation adds conviction. Designed for 15-25 trades/year on 1d to minimize fee drag.
# Works in both bull and bear markets by trading breakouts in the direction of the higher timeframe trend.

name = "1d_Donchian20_1wEMA34_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for trend filter (HTF = 1w as per experiment)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian channels (20-period) for breakout signals
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient warmup for Donchian
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1w_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels using data up to current bar (no look-ahead)
        highest_high = np.max(high[i-19:i+1])  # 20-period high
        lowest_low = np.min(low[i-19:i+1])     # 20-period low
        
        # Volume confirmation: 20-period EMA
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high
        breakout_down = close[i] < lowest_low
        
        if position == 0:
            # Long: breakout above upper Donchian in 1w uptrend with volume spike
            if breakout_up and ema_34_1w_aligned[i] > close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: breakout below lower Donchian in 1w downtrend with volume spike
            elif breakout_down and ema_34_1w_aligned[i] < close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches middle of Donchian channel or loses 1w uptrend
            donchian_mid = (highest_high + lowest_low) / 2.0
            if close[i] <= donchian_mid or ema_34_1w_aligned[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches middle of Donchian channel or loses 1w downtrend
            donchian_mid = (highest_high + lowest_low) / 2.0
            if close[i] >= donchian_mid or ema_34_1w_aligned[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals