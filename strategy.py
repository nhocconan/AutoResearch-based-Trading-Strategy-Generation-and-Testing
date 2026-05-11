# The strategy is a 4-hour breakout strategy based on a 12-hour trend filter and volume confirmation.
# The hypothesis is that price will continue in the direction of the 12-hour trend after breaking
# a short-term range, with volume confirming the breakout's strength. This should work in both
# bull and bear markets by following the higher timeframe trend.
# The strategy uses a 20-period high/low for the breakout level, a 50-period EMA on 12h for trend,
# and a 20-period volume average for volume confirmation. It aims for infrequent, high-quality trades.

#!/usr/bin/env python3
name = "4h_Breakout_12Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 4h 20-period high/low for breakout levels
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            # Maintain current position if any
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge filter
        volume_surge = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: Break above 20-period high with volume and above 12h EMA50 (bullish trend)
            if (close[i] > high_max[i] and 
                volume_surge and 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below 20-period low with volume and below 12h EMA50 (bearish trend)
            elif (close[i] < low_min[i] and 
                  volume_surge and 
                  close[i] < ema_50_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to the opposite breakout level or trend fails
            if position == 1:
                # Exit long: price breaks below 20-period low or trend turns bearish
                if (close[i] < low_min[i]) or (close[i] < ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above 20-period high or trend turns bullish
                if (close[i] > high_max[i]) or (close[i] > ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals