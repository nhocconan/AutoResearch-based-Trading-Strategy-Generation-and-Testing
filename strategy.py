#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot (R3/S3) breakout with 1d trend filter (EMA34) and volume spike.
# Long when price breaks above R3 AND 1d EMA34 rising AND volume > 2x 20-period average.
# Short when price breaks below S3 AND 1d EMA34 falling AND volume > 2x 20-period average.
# Exit when price crosses back inside the Camarilla H-L range.
# This strategy targets volatility expansion at key pivot levels with trend alignment.
# Works in both bull and bear markets by following the 1d trend direction.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    
    # Calculate Camarilla levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Use previous bar's typical price for pivot calculation (no look-ahead)
    typical_price_prev = np.roll(typical_price, 1)
    typical_price_prev[0] = typical_price[0]  # first bar uses current
    
    # Camarilla levels based on previous day's range
    # R3 = close + (high - low) * 1.1/2
    # S3 = close - (high - low) * 1.1/2
    range_prev = np.roll(high - low, 1)
    range_prev[0] = high[0] - low[0]
    r3 = typical_price_prev + range_prev * 1.1 / 2.0
    s3 = typical_price_prev - range_prev * 1.1 / 2.0
    
    # H-L range for exit (using previous bar's high/low)
    h_prev = np.roll(high, 1)
    l_prev = np.roll(low, 1)
    h_prev[0] = high[0]
    l_prev[0] = low[0]
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(h_prev[i]) or np.isnan(l_prev[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or np.isnan(ema34_falling[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, 1d EMA34 rising, volume filter
            long_cond = (close[i] > r3[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price breaks below S3, 1d EMA34 falling, volume filter
            short_cond = (close[i] < s3[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back inside H-L range (below previous day's high)
            if close[i] < h_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back inside H-L range (above previous day's low)
            if close[i] > l_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals