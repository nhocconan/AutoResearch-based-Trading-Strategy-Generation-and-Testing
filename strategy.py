#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike
# Donchian channels from daily price action provide robust breakout zones.
# 1w EMA50 filter ensures alignment with the weekly trend to avoid counter-trend trades.
# Volume spike confirms institutional participation at these key levels.
# Designed for very low trade frequency (target: 7-25/year) to minimize fee drag on 1d timeframe.
# Works in both bull and bear markets by trading with the higher timeframe trend.

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1w data for Donchian, EMA, and volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w Donchian channels (based on previous week's OHLC)
    # Upper = max(high of last 20 weeks), Lower = min(low of last 20 weeks)
    # We calculate for the PREVIOUS week to avoid look-ahead
    prev_high = df_1w['high'].shift(1).rolling(window=20, min_periods=20).max().values
    prev_low = df_1w['low'].shift(1).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1w EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1w volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1w['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1w['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1w indicators to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1w, prev_high)
    lower_aligned = align_htf_to_ltf(prices, df_1w, prev_low)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1w, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_spike_aligned[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend direction
        is_uptrend = close[i] > ema_50_aligned[i]
        is_downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper channel in uptrend with volume spike
            # OR price breaks above upper channel + 0.5*ATR (strong breakout) regardless of trend
            if ((high[i] > upper_aligned[i] and is_uptrend and volume_spike_aligned[i]) or
                (high[i] > upper_aligned[i] * 1.005)):  # 0.5% buffer for strong breakout
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower channel in downtrend with volume spike
            # OR price breaks below lower channel - 0.5*ATR (strong breakout) regardless of trend
            elif ((low[i] < lower_aligned[i] and is_downtrend and volume_spike_aligned[i]) or
                  (low[i] < lower_aligned[i] * 0.995)):  # 0.5% buffer for strong breakout
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below lower channel (reversal) or hits upper channel * 1.02 (profit target)
            if low[i] < lower_aligned[i] or high[i] > upper_aligned[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above upper channel (reversal) or hits lower channel * 0.98 (profit target)
            if high[i] > upper_aligned[i] or low[i] < lower_aligned[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals