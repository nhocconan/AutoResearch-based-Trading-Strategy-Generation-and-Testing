#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above R3 AND price > 1d EMA34 AND volume > 1.5x 20-period average.
# Short when price breaks below S3 AND price < 1d EMA34 AND volume > 1.5x 20-period average.
# Exit when price returns to center (P) or trend reverses.
# Uses Camarilla levels from daily OHLC for institutional support/resistance.
# Designed for 4h timeframe with controlled trade frequency (target: 20-40/year).
# Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend).

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Need previous day's data - we'll calculate for each bar using prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Align daily OHLC to 4h timeframe
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Camarilla levels for each bar using previous day's OHLC
    # R3 = close + 1.1*(high - low)
    # S3 = close - 1.1*(high - low)
    # P = (high + low + close)/3
    diff = high_1d_aligned - low_1d_aligned
    r3 = close_1d_aligned + 1.1 * diff
    s3 = close_1d_aligned - 1.1 * diff
    p = (high_1d_aligned + low_1d_aligned + close_1d_aligned) / 3.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(p[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 AND price > 1d EMA34 AND volume filter
            long_cond = (close[i] > r3[i]) and (close[i] > ema34_1d_aligned[i]) and volume_filter[i]
            # Short conditions: price breaks below S3 AND price < 1d EMA34 AND volume filter
            short_cond = (close[i] < s3[i]) and (close[i] < ema34_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot (P) OR price < 1d EMA34 (trend reversal)
            if close[i] <= p[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to pivot (P) OR price > 1d EMA34 (trend reversal)
            if close[i] >= p[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals