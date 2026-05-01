#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d EMA34 trend filter + volume confirmation
# Uses 1d EMA34 to filter trend direction: price > EMA34 = bullish bias (longs only), price < EMA34 = bearish bias (shorts only)
# Donchian(20) breakout on 6f timeframe: long on break above 20-period high, short on break below 20-period low
# Volume confirmation: current volume > 1.5 * 20-period EMA of volume
# Designed for low frequency (50-150 trades over 4 years) with clear structure and trend alignment

name = "6h_Donchian20_1dEMA34_VolumeTrend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 6h Donchian(20) channels
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume spike filter: volume > 1.5 * 20-period EMA
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(34, 20)  # Need EMA34 and Donchian20
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA34
        bullish_bias = close[i] > ema34_1d_aligned[i]
        bearish_bias = close[i] < ema34_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Donchian breakout above 20-period high + volume spike + bullish bias
            if high[i] > high_max_20[i-1] and volume_spike[i] and bullish_bias:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout below 20-period low + volume spike + bearish bias
            elif low[i] < low_min_20[i-1] and volume_spike[i] and bearish_bias:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Donchian breakdown below 20-period low OR loss of bullish bias
            if low[i] < low_min_20[i-1] or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Donchian breakout above 20-period high OR loss of bearish bias
            if high[i] > high_max_20[i-1] or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals