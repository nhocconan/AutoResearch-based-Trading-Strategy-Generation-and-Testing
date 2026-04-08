# Hypothesis: On daily timeframe, combine 20-day Donchian breakout with weekly trend filter and volume confirmation to capture sustained trends while avoiding false breakouts in choppy markets. The weekly trend filter ensures we only trade in the direction of the higher timeframe trend, reducing whipsaws. Volume confirmation adds confirmation of institutional interest. This approach should work in both bull and bear markets by capturing strong directional moves while avoiding chop.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # 1d data for Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume filter: current volume > 1.8x 20-period average (to avoid low-volume breakouts)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below 20-day low or trend reverses
            if close[i] < lowest_20[i] or close[i] < ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 20-day high or trend reverses
            if close[i] > highest_20[i] or close[i] > ema_50_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above 20-day high + uptrend on weekly + volume spike
            if (close[i] > highest_20[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                vol_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below 20-day low + downtrend on weekly + volume spike
            elif (close[i] < lowest_20[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  vol_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals