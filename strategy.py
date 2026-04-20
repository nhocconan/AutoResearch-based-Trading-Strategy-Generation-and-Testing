# 1d_1w_Pivot_R2S2_Breakout_VolumeTrend
# Hypothesis: On 1d timeframe, trade breakouts from 1w-derived R2/S2 levels with volume confirmation and 1w EMA trend filter.
# R2/S2 represent moderate breakout levels, balancing sensitivity and false signals.
# Uses 1w EMA34 to filter trades in trending markets, targeting 15-30 trades per year.
# Works in both bull and bear markets by aligning with 1w trend direction.

name = "1d_1w_Pivot_R2S2_Breakout_VolumeTrend"
timeframe = "1d"
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
    
    # Get 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w R2 and S2 levels using previous week's data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point and range
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Camarilla levels: R2 and S2
    s2_1w = close_1w - (range_1w * 1.1 / 4)
    r2_1w = close_1w + (range_1w * 1.1 / 4)
    
    # 1w EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w levels to 1d timeframe
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume average for spike detection (20-period)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(s2_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(ema_34_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above R2, volume spike, and price above 1w EMA34 (uptrend)
            if (close[i] > r2_aligned[i] * 1.003 and 
                volume[i] > 2.0 * volume_ma[i] and
                close[i] > ema_34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below S2, volume spike, and price below 1w EMA34 (downtrend)
            elif (close[i] < s2_aligned[i] * 0.997 and 
                  volume[i] > 2.0 * volume_ma[i] and
                  close[i] < ema_34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price below S2 or trend reversal (below EMA34)
            if close[i] < s2_aligned[i] * 0.997 or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price above R2 or trend reversal (above EMA34)
            if close[i] > r2_aligned[i] * 1.003 or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals