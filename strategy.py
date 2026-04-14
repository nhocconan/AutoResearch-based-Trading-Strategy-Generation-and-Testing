#!/usr/bin/env python3
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
    
    # Get daily data for pivot points and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ATR(14) for volatility filter
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    tr1 = high_1d_series - low_1d_series
    tr2 = abs(high_1d_series - close_1d_series.shift(1))
    tr3 = abs(low_1d_series - close_1d_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14_1d = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate daily pivot points (using prior day's OHLC)
    prev_day_high = np.roll(high_1d, 1)
    prev_day_low = np.roll(low_1d, 1)
    prev_day_close = np.roll(close_1d, 1)
    prev_day_high[0] = np.nan
    prev_day_low[0] = np.nan
    prev_day_close[0] = np.nan
    
    # Daily pivot point
    pp = (prev_day_high + prev_day_low + prev_day_close) / 3
    # Daily resistance and support levels
    r1 = 2 * pp - prev_day_low
    s1 = 2 * pp - prev_day_high
    r2 = pp + (prev_day_high - prev_day_low)
    s2 = pp - (prev_day_high - prev_day_low)
    
    # Align daily pivot levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Volume confirmation: volume > 1.3x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(20, 14)  # for 20-period volume average and 14-period ATR
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(atr_14_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above daily R2 with volume and volatility filter
            if (price > r2_aligned[i] and 
                vol > 1.3 * avg_vol[i] and atr_14_aligned[i] > 0):
                position = 1
                signals[i] = position_size
            # Short: price breaks below daily S2 with volume and volatility filter
            elif (price < s2_aligned[i] and 
                  vol > 1.3 * avg_vol[i] and atr_14_aligned[i] > 0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below daily S1
            if price < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above daily R1
            if price > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Daily_Pivot_Volume_Filter_v3"
timeframe = "12h"
leverage = 1.0