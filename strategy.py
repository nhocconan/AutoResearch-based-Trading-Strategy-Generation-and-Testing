#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Williams Alligator identifies trend via smoothed medians (Jaw/Teeth/Lips). 
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades.
# Volume spike confirms institutional participation. Designed for 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via Alligator uptrend continuation and in bear markets via downtrend shorts.

name = "12h_WilliamsAlligator_1dEMA34_VolumeSpike"
timeframe = "12h"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 12h data (smoothed medians)
    median = (high + low) / 2
    jaw = pd.Series(median).rolling(window=13, min_periods=13).mean().shift(8).values  # 13-period, 8-bar shift
    teeth = pd.Series(median).rolling(window=8, min_periods=8).mean().shift(5).values   # 8-period, 5-bar shift
    lips = pd.Series(median).rolling(window=5, min_periods=5).mean().shift(3).values    # 5-period, 3-bar shift
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from sufficient lookback for Alligator
    start_idx = max(13, 8)  # jaw needs 13+8=21 bars
    
    for i in range(start_idx, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: 20-period EMA on 12h
        if i >= 19:
            vol_ema_20 = pd.Series(volume[i-19:i+1]).ewm(span=20, adjust=False, min_periods=1).mean().iloc[-1]
        else:
            vol_ema_20 = volume[i]
        volume_spike = volume[i] > (1.5 * vol_ema_20)
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (Alligator uptrend) + price above Lips + 1d uptrend + volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > lips[i] and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (Alligator downtrend) + price below Lips + 1d downtrend + volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < lips[i] and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator trend reverses or loses 1d uptrend
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator trend reverses or loses 1d downtrend
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals