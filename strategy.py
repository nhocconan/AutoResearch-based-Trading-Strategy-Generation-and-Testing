# 1d_Camarilla_R3S3_Breakout_1wTrend_Volume
# Strategy: 1d timeframe with 1h resolution, using 1h for entry timing and 1w for trend filter.
# Long when price breaks above weekly R3 AND 1w EMA34 rising AND volume > 2x 20-period average.
# Short when price breaks below weekly S3 AND 1w EMA34 falling AND volume > 2x 20-period average.
# Exit when price crosses back inside weekly H3-L3 range.
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1w trend direction.

name = "1d_Camarilla_R3S3_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla R3 and S3 from previous week's OHLC
    prev_close = df_1w['close'].shift(1).values
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    
    # Weekly Camarilla R3 and S3
    R3 = prev_close + 1.1 * (prev_high - prev_low) / 6
    S3 = prev_close - 1.1 * (prev_high - prev_low) / 6
    
    # Align weekly levels to 1d timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    
    # 1w EMA34 for trend filter
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # 1w EMA34 direction
    ema34_rising = np.zeros_like(ema34_1w_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1w_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1w_aligned[1:] > ema34_1w_aligned[:-1]
    ema34_falling[1:] = ema34_1w_aligned[1:] < ema34_1w_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Sufficient warmup for EMA34 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, 1w EMA34 rising, volume filter
            long_cond = (close[i] > R3_aligned[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price breaks below S3, 1w EMA34 falling, volume filter
            short_cond = (close[i] < S3_aligned[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below S3
            if close[i] < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above R3
            if close[i] > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals