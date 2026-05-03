#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume spike
# Donchian channels provide robust structure-based breakouts that work in both bull and bear markets.
# Breakout above upper channel or below lower channel with 1d trend alignment captures momentum.
# Volume spike confirms institutional participation. Designed for 12-37 trades/year on 12h to minimize fee drag.
# Works in bull markets via buying upper breakouts in uptrends and bear markets via selling lower breakouts in downtrends.

name = "12h_Donchian20_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Donchian channels (20-period)
    if len(df_1d) >= 20:
        donchian_high = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
        donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
        donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    else:
        donchian_high_aligned = np.full(n, np.nan)
        donchian_low_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Need at least 20 bars for Donchian calculation
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or not in_session[i]):
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
        
        # Donchian breakout conditions
        breakout_up = high[i] > donchian_high_aligned[i]
        breakout_down = low[i] < donchian_low_aligned[i]
        
        if position == 0:
            # Long: bullish breakout above upper Donchian in 1d uptrend with volume spike
            if breakout_up and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish breakdown below lower Donchian in 1d downtrend with volume spike
            elif breakout_down and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to lower Donchian or loses 1d uptrend
            if low[i] < donchian_low_aligned[i] or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to upper Donchian or loses 1d downtrend
            if high[i] > donchian_high_aligned[i] or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals