#!/usr/bin/env python3
"""
1d Donchian(20) Breakout with Volume Spike and 1w EMA50 Trend Filter
Hypothesis: Donchian channel breakouts capture strong momentum moves. Volume confirmation filters false breakouts, while the 1w EMA50 ensures alignment with the weekly trend, improving performance in both bull and bear markets. Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 10-25 trades/year on 1d.
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
    
    # Get 1w data for EMA trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian(20) channels from 1d OHLC
    # Need 20 days of data for channel calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Upper channel: 20-day high, Lower channel: 20-day low
    upper_channel = pd.Series(df_1d['high'].values).rolling(window=20, min_periods=20).max().values
    lower_channel = pd.Series(df_1d['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d (use previous day's levels for current day's trading)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Donchian calculation and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        upper_level = upper_aligned[i]
        lower_level = lower_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above upper channel AND volume spike AND price > 1w EMA50 (uptrend)
            long_entry = (curr_close > upper_level) and vol_spike and (curr_close > ema_trend)
            # Short: price breaks below lower channel AND volume spike AND price < 1w EMA50 (downtrend)
            short_entry = (curr_close < lower_level) and vol_spike and (curr_close < ema_trend)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below lower channel (reversal) OR price < 1w EMA50 (trend change)
            if (curr_close < lower_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above upper channel (reversal) OR price > 1w EMA50 (trend change)
            if (curr_close > upper_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_VolumeSpike_1wEMA50_Trend"
timeframe = "1d"
leverage = 1.0