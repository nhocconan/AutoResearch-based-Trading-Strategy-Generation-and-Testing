#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h momentum with 4h trend filter and volume confirmation
# Long when price breaks above 1h high of 20-period AND 4h EMA(50) rising AND volume > 1.5x average
# Short when price breaks below 1h low of 20-period AND 4h EMA(50) falling AND volume > 1.5x average
# Exit when price crosses 10-period EMA in opposite direction
# Uses 4h for trend direction to reduce whipsaw, 1h for precise entry timing
# Target: 60-150 total trades over 4 years (15-37/year) for optimal 1h performance

name = "1h_momentum_4h_trend_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1h indicators: 20-period high/low for breakout, 10-period EMA for exit
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    highest_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    lowest_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    ema_10 = close_series.ewm(span=10, min_periods=10, adjust=False).mean().values
    
    # 4h trend filter: EMA(50) slope
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_50_4h_slope = np.diff(ema_50_4h_aligned, prepend=ema_50_4h_aligned[0])
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) or 
            np.isnan(ema_10[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: price crosses 10-period EMA in opposite direction
        if position == 1:  # long position
            if close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            if close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with trend filter and volume confirmation
            # Long: price breaks above 20-period high AND 4h EMA rising AND volume confirmation
            if (close[i] > highest_high_20[i] and close[i-1] <= highest_high_20[i-1] and
                ema_50_4h_slope[i] > 0 and volume[i] > volume_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 20-period low AND 4h EMA falling AND volume confirmation
            elif (close[i] < lowest_low_20[i] and close[i-1] >= lowest_low_20[i-1] and
                  ema_50_4h_slope[i] < 0 and volume[i] > volume_threshold[i]):
                signals[i] = -0.20
                position = -1
    
    return signals