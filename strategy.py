#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d trend filter + volume confirmation
# - Donchian channel breakout provides clear entry/exit signals
# - 1d EMA50 trend filter ensures trades align with higher timeframe trend
# - Volume spike (>2x 20-period average) confirms breakout strength
# - Works in both bull/bear markets by using 1d trend filter
# - Target: 20-50 trades/year to minimize fee drag on 4h timeframe

name = "4h_Donchian_20_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian channel (20-period) on 4h data
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_max20[i]) or np.isnan(low_min20[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper + 1d uptrend + volume spike
            long_cond = (close[i] > high_max20[i] and 
                        ema_50_1d_aligned[i] > close_1d[-1] if i == len(prices)-1 else ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below Donchian lower + 1d downtrend + volume spike
            short_cond = (close[i] < low_min20[i] and 
                         ema_50_1d_aligned[i] < close_1d[-1] if i == len(prices)-1 else ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1] and
                         volume_spike[i])
            
            # Simplified trend check (avoiding index issues)
            if i > 0:
                trend_up = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
                trend_down = ema_50_1d_aligned[i] < ema_50_1d_aligned[i-1]
            else:
                trend_up = trend_down = False
            
            long_cond = (close[i] > high_max20[i] and trend_up and volume_spike[i])
            short_cond = (close[i] < low_min20[i] and trend_down and volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian lower
            if close[i] < low_min20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian upper
            if close[i] > high_max20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals