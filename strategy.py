#!/usr/bin/env python3
# 12h_1w_ema_trend_volume_v1
# Hypothesis: Trade 12-hour EMA trend with 1-week EMA filter and volume confirmation.
# Enter long when price is above 12h EMA(20) and 1w EMA(20) is rising with volume > 1.5x average.
# Enter short when price is below 12h EMA(20) and 1w EMA(20) is falling with volume > 1.5x average.
# Exit when price crosses the 12h EMA(20) or 1w EMA(20) flattens.
# Trend filter avoids whipsaws in sideways markets. Volume confirms trend strength.
# Target: 12-30 trades/year with strict entry conditions to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_ema_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-week EMA(20) trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(20) for 1w
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate EMA slope (rising/falling) for 1w
    ema_slope_1w = np.diff(ema_20_1w, prepend=ema_20_1w[0])
    
    # Pad arrays to match original length
    ema_20_1w_pad = np.zeros(len(close_1w))
    ema_slope_1w_pad = np.zeros(len(close_1w))
    ema_20_1w_pad[:] = ema_20_1w
    ema_slope_1w_pad[:] = ema_slope_1w
    
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w_pad)
    ema_slope_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_1w_pad)
    
    # 12h EMA(20)
    ema_20_12h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: 12h volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 40  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(ema_slope_1w_aligned[i]) or 
            np.isnan(ema_20_12h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: Price below 12h EMA(20) OR 1w EMA slope <= 0 (flattening/falling)
            if close[i] < ema_20_12h[i] or ema_slope_1w_aligned[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above 12h EMA(20) OR 1w EMA slope >= 0 (flattening/rising)
            if close[i] > ema_20_12h[i] or ema_slope_1w_aligned[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above 12h EMA(20) AND 1w EMA rising AND volume surge
            if (close[i] > ema_20_12h[i] and  
                ema_slope_1w_aligned[i] > 0 and 
                vol_surge):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below 12h EMA(20) AND 1w EMA falling AND volume surge
            elif (close[i] < ema_20_12h[i] and 
                  ema_slope_1w_aligned[i] < 0 and 
                  vol_surge):
                position = -1
                signals[i] = -0.25
    
    return signals