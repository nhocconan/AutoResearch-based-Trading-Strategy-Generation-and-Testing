#!/usr/bin/env python3
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
    
    # Get weekly data for trend direction
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly SMA(40) for trend filter
    sma_40_1w = pd.Series(close_1w).rolling(window=40, min_periods=40).mean().values
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points using prior day's OHLC
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Pivot point: (H + L + C) / 3
    pp = (prev_high + prev_low + prev_close) / 3
    # Resistance and support levels
    r1 = 2 * pp - prev_low
    s1 = 2 * pp - prev_high
    r2 = pp + (high_1d - low_1d)
    s2 = pp - (high_1d - low_1d)
    r3 = r1 + (high_1d - low_1d)
    s3 = s1 - (high_1d - low_1d)
    
    # Align weekly SMA and daily pivot levels to 6h timeframe
    sma_40_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_40_1w)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for weekly SMA and volume calculation
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(sma_40_1w_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long: price breaks above R2 with volume confirmation AND weekly uptrend
            if price > r2_aligned[i] and vol > 1.5 * avg_vol[i] and close_1w[-1] > sma_40_1w_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below S2 with volume confirmation AND weekly downtrend
            elif price < s2_aligned[i] and vol > 1.5 * avg_vol[i] and close_1w[-1] < sma_40_1w_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below S1 or weekly trend turns down
            if price < s1_aligned[i] or close_1w[-1] < sma_40_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above R1 or weekly trend turns up
            if price > r1_aligned[i] or close_1w[-1] > sma_40_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_1w_Pivot_Trend_Filter"
timeframe = "6h"
leverage = 1.0