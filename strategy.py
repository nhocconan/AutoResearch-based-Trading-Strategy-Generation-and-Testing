#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above 20-period high AND close > EMA50(12h) AND volume > 1.8x 20-period average
# Short when price breaks below 20-period low AND close < EMA50(12h) AND volume > 1.8x 20-period average
# Exit when price crosses back to 10-period midpoint OR EMA50(12h) trend flips
# Donchian channels provide clear structure; 12h EMA filters counter-trend moves; volume confirms validity
# Target: 20-50 trades/year per symbol (80-200 total over 4 years) for 4h timeframe
# Discrete sizing (0.25) to limit fee drag

name = "4h_Donchian20_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h close for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period) on primary timeframe
    if len(high) >= 20:
        # Rolling max/min for Donchian channels
        high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
        low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Midpoint for exit
        donchian_mid = (high_roll_max + low_roll_min) / 2
    else:
        high_roll_max = np.full(n, np.nan)
        low_roll_min = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.8x 20-period average
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.8 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(high_roll_max[i]) or 
            np.isnan(low_roll_min[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above 20-period high AND close > EMA50(12h) AND volume spike
            if (close[i] > high_roll_max[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below 20-period low AND close < EMA50(12h) AND volume spike
            elif (close[i] < low_roll_min[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below midpoint OR close < EMA50(12h) (trend flip)
            if (close[i] < donchian_mid[i] or 
                close[i] < ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above midpoint OR close > EMA50(12h) (trend flip)
            if (close[i] > donchian_mid[i] or 
                close[i] > ema_50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals