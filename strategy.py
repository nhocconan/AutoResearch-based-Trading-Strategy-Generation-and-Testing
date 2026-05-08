# 12h_Camarilla_R3S3_1dVolume_1dEMA34
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and 1d EMA34 trend filter.
# Long when price breaks above Camarilla R3 AND 1d volume > 2.0x 20-period average AND price > 1d EMA34.
# Short when price breaks below Camarilla S3 AND 1d volume > 2.0x 20-period average AND price < 1d EMA34.
# Exit when price crosses back below/above 1d EMA34 (trend-based exit).
# Uses R3/S3 levels (stronger reversal) and volume filter to reduce trades, targeting 50-150 total trades over 4 years.
# Timeframe: 12h (lower frequency reduces fee drift). Works in bull via trend-following breaks, in bear via mean-reversion at strong S3/R3 levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3S3_1dVolume_1dEMA34"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1d data for Camarilla, volume, and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels using previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla levels: R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    camarilla_range = prev_high - prev_low
    r3 = prev_close + camarilla_range * 1.1 / 4
    s3 = prev_close - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe (using previous day's levels)
    r3_12h = align_htf_to_ltf(prices, df_1d, r3)
    s3_12h = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d volume filter: current volume > 2.0x 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20)
    volume_filter = vol_1d > (2.0 * vol_ma20_aligned)
    
    # 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r3_12h[i]) or np.isnan(s3_12h[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, volume spike, above 1d EMA34
            long_cond = (close[i] > r3_12h[i]) and volume_filter[i] and (close[i] > ema34_1d_aligned[i])
            # Short conditions: price breaks below S3, volume spike, below 1d EMA34
            short_cond = (close[i] < s3_12h[i]) and volume_filter[i] and (close[i] < ema34_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1d EMA34 (trend change)
            if close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 1d EMA34 (trend change)
            if close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals