#!/usr/bin/env python3
"""
6h_Donchian20_VolumeSpike_1dTrend_Filter_v1
Hypothesis: 6h Donchian(20) breakout with volume spike confirmation and 1d EMA50 trend filter. 
Long when price breaks above upper Donchian channel + volume > 1.5x 20-period average + close > 1d EMA50.
Short when price breaks below lower Donchian channel + volume > 1.5x 20-period average + close < 1d EMA50.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 50-150 trades over 4 years by requiring 
three-way confluence (breakout, volume, trend). Works in bull/bear via trend filter - only takes trades 
in direction of higher timeframe trend, reducing false breakouts in choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period)
    donchian_period = 20
    upper_channel = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower_channel = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate volume spike filter (volume > 1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Calculate 1d EMA50 for HTF trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    htf_trend = np.where(close > ema_50_1d_aligned, 1, -1)  # 1 = uptrend, -1 = downtrend
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 20 for volume MA, 50 for EMA)
    start_idx = max(20, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Breakout conditions
        bullish_breakout = close[i] > upper_channel[i-1]  # Price breaks above previous upper channel
        bearish_breakout = close[i] < lower_channel[i-1]  # Price breaks below previous lower channel
        
        # Entry logic
        if bullish_breakout and volume_spike[i] and htf_trend[i] == 1:
            # Long: bullish breakout + volume spike + uptrend HTF
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif bearish_breakout and volume_spike[i] and htf_trend[i] == -1:
            # Short: bearish breakout + volume spike + downtrend HTF
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        else:
            # Exit conditions: reverse signal or loss of volume spike/trend alignment
            if position == 1 and (not volume_spike[i] or htf_trend[i] != 1 or bearish_breakout):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (not volume_spike[i] or htf_trend[i] != -1 or bullish_breakout):
                signals[i] = 0.0
                position = 0
            else:
                # Hold current position
                if position == 0:
                    signals[i] = 0.0
                elif position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_VolumeSpike_1dTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0