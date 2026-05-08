# 12h_Camarilla_R1S1_1dVolume_Trend
# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume confirmation and 1d EMA trend filter.
# Long when price breaks above Camarilla R1 AND 1d volume > 1.8x 20-period average AND price > 1d EMA(50).
# Short when price breaks below Camarilla S1 AND 1d volume > 1.8x 20-period average AND price < 1d EMA(50).
# Exit when price crosses back below/above EMA(50) (trend-based exit).
# Target: 50-150 total trades over 4 years (12-37/year) with controlled frequency.
# Works in both bull and bear markets: breakouts capture trends, volume confirms strength,
# EMA filter avoids counter-trend trades, Camarilla levels provide precise reversal points.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1S1_1dVolume_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla levels, volume filter, and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (using previous day's levels)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d volume filter: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (1.8 * vol_ma20)
    volume_filter = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    # 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R1, volume spike, above 1d EMA50
            long_cond = (close[i] > r1_12h[i]) and volume_filter[i] and (close[i] > ema_50_1d_aligned[i])
            # Short conditions: price breaks below S1, volume spike, below 1d EMA50
            short_cond = (close[i] < s1_12h[i]) and volume_filter[i] and (close[i] < ema_50_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1d EMA50 (trend change)
            if close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 1d EMA50 (trend change)
            if close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals