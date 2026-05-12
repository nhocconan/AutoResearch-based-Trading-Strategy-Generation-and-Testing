# 4H_DONCHIAN_BREAKOUT_1D_TREND_VOLUME_CONFIRMATION
# Hypothesis: 4h Donchian breakout (20-period) with 1d EMA trend filter and volume spike confirmation.
# In uptrend (price > 1d EMA), go long on upper band breakout with volume > 1.5x average.
# In downtrend (price < 1d EMA), go short on lower band breakout with volume > 1.5x average.
# Exit on opposite band touch. Works in bull/bear via trend filter; volume reduces false breakouts.
# Target: 20-50 trades/year on 4h timeframe.

name = "4H_DONCHIAN_BREAKOUT_1D_TREND_VOLUME_CONFIRMATION"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d EMA for trend filter (34-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if trend data not ready
        if np.isnan(ema34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Uptrend + upper band breakout + volume spike
            if (close[i] > ema34_1d_aligned[i] and 
                high[i] > high_max[i-1] and 
                volume[i] > 1.5 * vol_ma[i-1]):
                signals[i] = 0.25
                position = 1
            # SHORT: Downtrend + lower band breakout + volume spike
            elif (close[i] < ema34_1d_aligned[i] and 
                  low[i] < low_min[i-1] and 
                  volume[i] > 1.5 * vol_ma[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Touch or cross lower band
            if low[i] <= low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Touch or cross upper band
            if high[i] >= high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals