#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout with 12h Trend and Volume Spike
# - Uses Donchian channel from 4h timeframe (20-period high/low)
# - Breakout above upper band with 12h uptrend or below lower band with 12h downtrend
# - Volume spike confirms breakout strength
# - Works in bull/bear by using 12h trend filter to avoid counter-trend trades
# - Target: 20-40 trades/year to minimize fee drag on 4h timeframe

name = "4h_DonchianBreakout_12hTrend_Volume"
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
    
    # Donchian channel (20-period high/low) on 4h
    donchian_period = 20
    high_max = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    low_min = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or \
           np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_spike[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band with 12h uptrend + volume spike
            long_cond = (close[i] > high_max[i] and 
                        ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below lower Donchian band with 12h downtrend + volume spike
            short_cond = (close[i] < low_min[i] and 
                         ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below lower Donchian band
            if close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above upper Donchian band
            if close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals