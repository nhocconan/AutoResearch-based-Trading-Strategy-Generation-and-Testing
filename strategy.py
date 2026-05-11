# 12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Camarilla R1/S1 breakout on 12h timeframe with 1d trend filter (EMA34) and volume confirmation.
# Uses tighter entry (R1/S1 instead of R3/S3) for better win rate, with volume filter to avoid false breakouts.
# Designed to work in both bull and bear markets by following 1d trend direction.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within fee drag limits.
# Uses discrete position sizing (0.25) to minimize churn.

#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels from previous 1d bar's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous 1d bar's range
    range_1d = high_1d - low_1d
    
    # Calculate Camarilla R1 and S1 levels (tighter than R3/S3)
    camarilla_r1 = close_1d + (range_1d * 1.1 / 6)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 6)
    
    # Align Camarilla levels to 12h timeframe (using previous 1d bar's values)
    r1_12h = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=34, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: current volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 AND above 1d EMA34 (uptrend) AND volume surge
            if close[i] > r1_12h[i] and close[i] > ema_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND below 1d EMA34 (downtrend) AND volume surge
            elif close[i] < s1_12h[i] and close[i] < ema_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below S1 OR below 1d EMA34 (trend change)
            if close[i] < s1_12h[i] or close[i] < ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above R1 OR above 1d EMA34 (trend change)
            if close[i] > r1_12h[i] or close[i] > ema_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals