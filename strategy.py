#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: On daily timeframe, Donchian(20) breakouts with 1-week EMA50 trend filter and volume confirmation (>2x 20-day avg) capture sustained trends in both bull and bear markets. Weekly EMA50 provides robust trend filter to avoid false breakouts, while volume spike confirms institutional participation. Targets 15-25 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) channels on daily
    lookback = 20
    upper_channel = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lower_channel = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 2x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for calculations
    start_idx = max(lookback, 50, 20)  # Donchian20, weekly EMA50, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned values
        ema_50_val = ema_50_aligned[i]
        upper_val = upper_channel[i]
        lower_val = lower_channel[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 2x 20-day average
        volume_confirmed = vol_val > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with uptrend (close > EMA50) and volume confirmation
            long_signal = (high_val > upper_val) and (close_val > ema_50_val) and volume_confirmed
            # Short: price breaks below lower Donchian channel with downtrend (close < EMA50) and volume confirmation
            short_signal = (low_val < lower_val) and (close_val < ema_50_val) and volume_confirmed
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price closes below lower Donchian channel
            if close_val < lower_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price closes above upper Donchian channel
            if close_val > upper_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0