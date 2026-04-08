# 4h Donchian Breakout with 1d Trend and Volume Confirmation
# Hypothesis: 4-hour Donchian(20) breakouts aligned with daily EMA(50) trend
# and volume > 2x 20-period average capture strong momentum in both bull and bear markets.
# Breakouts above upper band or below lower band trigger entries; exits on opposite band touch.
# Designed for 4h timeframe to achieve 20-40 trades/year with clear trend following logic.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_trend_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 4h Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA(50) for trend filter
    ema_50 = df_1d['close'].ewm(span=50, adjust=False).mean().values
    ema_50_4h = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume filter (>2x 20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_4h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price touches or crosses below lower Donchian band
            if close[i] <= low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.30
                
        elif position == -1:  # Short position
            # Exit: price touches or crosses above upper Donchian band
            if close[i] >= high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.30
        else:  # Flat, look for entry
            # Breakout long above upper band with trend alignment
            if (close[i] > high_20[i] and 
                close[i] > ema_50_4h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.30
            # Breakout short below lower band with trend alignment
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_4h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.30
    
    return signals