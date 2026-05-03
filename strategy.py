#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
# Donchian breakouts capture momentum with clear entry/exit levels.
# 1d EMA34 ensures trades align with the daily trend to avoid counter-trend whipsaws.
# Volume spike (>1.5x 20-period EMA) confirms institutional participation.
# Designed for low trade frequency (target: 12-37/year) on 12h timeframe to minimize fee drag.
# Works in bull markets via breakouts with trend, and in bear markets via short breakouts against trend (but filtered by EMA to reduce false signals).

name = "12h_Donchian20_1dEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels from last 20 periods (lookback window ends at i-1)
        if i >= 20:
            lookback_high = np.max(high[i-20:i])
            lookback_low = np.min(low[i-20:i])
        else:
            # Not enough lookback data
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend direction
        is_uptrend = close[i] > ema_34_aligned[i]
        is_downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian in uptrend with volume confirmation
            # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
            vol_ema_20 = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else volume[i]
            volume_confirmed = volume[i] > (1.5 * vol_ema_20)
            
            if high[i] > lookback_high and is_uptrend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian in downtrend with volume confirmation
            elif low[i] < lookback_low and is_downtrend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below lower Donchian (reversal) or hits upper Donchian (profit target)
            if low[i] < lookback_low or high[i] > lookback_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above upper Donchian (reversal) or hits lower Donchian (profit target)
            if high[i] > lookback_high or low[i] < lookback_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals