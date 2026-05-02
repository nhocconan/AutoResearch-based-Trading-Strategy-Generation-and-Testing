#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# Donchian channel breakouts capture strong momentum moves. 1d EMA50 ensures trades align with
# daily trend to avoid counter-trend whipsaws. Volume confirmation filters low-conviction breakouts.
# Works in bull markets (buying upper breakouts in uptrend) and bear markets
# (selling lower breakdowns in downtrend) by only taking trades in direction of 1d EMA50.
# Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.

name = "4h_Donchian20_Breakout_1dEMA50_Volume"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channel on 4h data
    period_donchian = 20
    high_roll = pd.Series(high).rolling(window=period_donchian, min_periods=period_donchian).max().values
    low_roll = pd.Series(low).rolling(window=period_donchian, min_periods=period_donchian).min().values
    upper_channel = high_roll
    lower_channel = low_roll
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = period_donchian
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above upper Donchian channel with volume spike AND price > 1d EMA50 (bullish trend)
            if (close[i] > upper_channel[i] and 
                volume_spike[i] and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below lower Donchian channel with volume spike AND price < 1d EMA50 (bearish trend)
            elif (close[i] < lower_channel[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below lower Donchian channel (channel break) OR price below 1d EMA50 (trend change)
            if close[i] < lower_channel[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above upper Donchian channel (channel break) OR price above 1d EMA50 (trend change)
            if close[i] > upper_channel[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals