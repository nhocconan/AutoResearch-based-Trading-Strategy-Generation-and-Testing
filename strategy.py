#!/usr/bin/env python3
# 6h_camarilla_pivot_volume_v1
# Hypothesis: 6h strategy using 1d Camarilla pivot levels with volume confirmation and 12h trend filter.
# Long: Price breaks above R3 with volume > 2x 20-period average and 12h close > 12h EMA20.
# Short: Price breaks below S3 with volume > 2x 20-period average and 12h close < 12h EMA20.
# Exit: Price returns to pivot point (PP) for both long and short.
# Uses 12h EMA20 for trend filter: only long when 12h close > 12h EMA20, only short when 12h close < 12h EMA20.
# Target: 12-30 trades/year to minimize fee drag while maintaining edge.
# Camarilla pivots work well in ranging markets (common in 2025 bear) and capture breakouts in trends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_prices = prices['open'].values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot Point (PP) = (High + Low + Close) / 3
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range = High - Low
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r1 = pp + (range_1d * 1.0 / 8.0)
    r2 = pp + (range_1d * 2.0 / 8.0)
    r3 = pp + (range_1d * 3.0 / 8.0)
    r4 = pp + (range_1d * 4.0 / 8.0)
    
    # Support levels
    s1 = pp - (range_1d * 1.0 / 8.0)
    s2 = pp - (range_1d * 2.0 / 8.0)
    s3 = pp - (range_1d * 3.0 / 8.0)
    s4 = pp - (range_1d * 4.0 / 8.0)
    
    # Align Camarilla levels to 6h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    open_12h = df_12h['open'].values
    
    # 12h EMA20 for trend filter
    close_12h_s = pd.Series(close_12h)
    ema_20_12h = close_12h_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align 12h EMA20 to 6h
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
    open_12h_aligned = align_htf_to_ltf(prices, df_12h, open_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after volume MA warmup
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i]) or
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(close_12h_aligned[i]) or
            np.isnan(open_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2x 20-period average
        volume_confirmed = volume[i] > 2.0 * volume_ma[i]
        # 12h trend filter: close > EMA20 for uptrend, < EMA20 for downtrend
        trend_12h_up = close_12h_aligned[i] > ema_20_12h_aligned[i]
        trend_12h_down = close_12h_aligned[i] < ema_20_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Price returns to pivot point (PP)
            if close[i] <= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price returns to pivot point (PP)
            if close[i] >= pp_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price breaks above R3 with volume and 12h uptrend
            if (close[i] > r3_aligned[i] and    # Break above R3
                volume_confirmed and           # Volume spike
                trend_12h_up):                 # 12h uptrend
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below S3 with volume and 12h downtrend
            elif (close[i] < s3_aligned[i] and  # Break below S3
                  volume_confirmed and          # Volume spike
                  trend_12h_down):              # 12h downtrend
                position = -1
                signals[i] = -0.25
    
    return signals