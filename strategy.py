#!/usr/bin/env python3
"""
#100850 - 1d_Donchian20_Breakout_1wTrend_Filter_VolumeSpike
Hypothesis: Donchian(20) breakout on 1d with 1w EMA20 trend filter and volume spike.
Works in bull (breakouts with trend) and bear (mean reversion to middle band).
Target: 8-20 trades/year to minimize fee drag. Uses 1d primary timeframe with 1w HTF.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1w EMA20 for trend filter
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Donchian(20) channels on 1d (using previous 20 days to avoid look-ahead)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume filter: volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema20_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: price breaks above upper Donchian, above 1w EMA20, volume spike
        if (close[i] > donchian_high[i] and 
            close[i] > ema20_1w_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: price breaks below lower Donchian, below 1w EMA20, volume spike
        elif (close[i] < donchian_low[i] and 
              close[i] < ema20_1w_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price returns to middle Donchian band (mean reversion)
        elif position == 1 and close[i] < donchian_mid[i]:
            signals[i] = 0.0
            position = 0
        elif position == -1 and close[i] > donchian_mid[i]:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_Filter_VolumeSpike"
timeframe = "1d"
leverage = 1.0