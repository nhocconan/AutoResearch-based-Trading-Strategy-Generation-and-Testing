# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# 4h_camarilla_pivot_1d_trend_volume_v2
# Hypothesis: On 4h timeframe, use Camarilla pivot levels from 1d chart with trend filter and volume confirmation.
# Long when price touches S3 support with volume > 1.5x average and 1d uptrend (close > EMA20).
# Short when price touches R3 resistance with volume > 1.5x average and 1d downtrend (close < EMA20).
# Exit when price crosses the daily pivot point (mean reversion) or volume drops below average.
# Uses tight entry conditions to limit trades and avoid fee drag. Works in both bull and bear markets via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Camarilla pivot levels
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot = (daily_high + daily_low + daily_close) / 3
    range_hl = daily_high - daily_low
    
    # Camarilla levels: S3, S2, S1, PP, R1, R2, R3
    s3 = daily_close - range_hl * 1.1 / 2
    s2 = daily_close - range_hl * 1.1 / 4
    s1 = daily_close - range_hl * 1.1 / 6
    pp = pivot
    r1 = daily_close + range_hl * 1.1 / 6
    r2 = daily_close + range_hl * 1.1 / 4
    r3 = daily_close + range_hl * 1.1 / 2
    
    # Align to 4h timeframe
    s3_4h = align_htf_to_ltf(prices, df_daily, s3)
    r3_4h = align_htf_to_ltf(prices, df_daily, r3)
    pp_4h = align_htf_to_ltf(prices, df_daily, pp)
    
    # Calculate 1d trend filter: EMA20
    daily_ema20 = pd.Series(daily_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    daily_ema20_4h = align_htf_to_ltf(prices, df_daily, daily_ema20)
    
    # Volume confirmation: 20-period average on 4h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(s3_4h[i]) or np.isnan(r3_4h[i]) or np.isnan(pp_4h[i]) or np.isnan(daily_ema20_4h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses above daily pivot (mean reversion) or volume drops below average
            if close[i] >= pp_4h[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses below daily pivot (mean reversion) or volume drops below average
            if close[i] <= pp_4h[i] or volume[i] < avg_volume[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Daily trend filter
            daily_uptrend = close[i] > daily_ema20_4h[i]
            daily_downtrend = close[i] < daily_ema20_4h[i]
            
            # Tolerance for touching levels (0.1% of price)
            tol = close[i] * 0.001
            
            # Long entry: price touches S3 support with volume and uptrend
            if abs(close[i] - s3_4h[i]) <= tol and volume_ok and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: price touches R3 resistance with volume and downtrend
            elif abs(close[i] - r3_4h[i]) <= tol and volume_ok and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals