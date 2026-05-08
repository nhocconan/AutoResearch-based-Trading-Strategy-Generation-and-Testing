# 12h Donchian20 Breakout with 1w Trend and Volume Spike
# - Uses 20-bar Donchian channel from 12h timeframe
# - Trend filter: 1w EMA50 (bullish when rising, bearish when falling)
# - Volume spike: current volume > 2.0x 20-period average
# - Breakout above upper band with uptrend + volume spike = long
# - Breakdown below lower band with downtrend + volume spike = short
# - Exit when price crosses middle band (mean of upper/lower)
# - Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe
# - Works in bull/bear by using 1w trend filter to avoid counter-trend trades

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Donchian20_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 12h Donchian channel (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_max
    donchian_lower = low_min
    donchian_middle = (donchian_upper + donchian_lower) / 2.0
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(donchian_middle[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper band with 1w uptrend + volume spike
            long_cond = (close[i] > donchian_upper[i] and 
                        ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and
                        volume_spike[i])
            
            # Short: price breaks below lower band with 1w downtrend + volume spike
            short_cond = (close[i] < donchian_lower[i] and 
                         ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and
                         volume_spike[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below middle band
            if close[i] < donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above middle band
            if close[i] > donchian_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3