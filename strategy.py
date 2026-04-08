#!/usr/bin/env python3
# 1d_camarilla_pivot_volume_filter_v1
# Hypothesis: Daily Camarilla pivot levels with volume confirmation and weekly trend filter.
# Long when: price touches S3 level, volume > 1.5x average, and weekly trend up (price > weekly EMA20).
# Short when: price touches R3 level, volume > 1.5x average, and weekly trend down (price < weekly EMA20).
# Exit when price crosses the daily VWAP or weekly trend reverses.
# Target: 10-25 trades/year to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_volume_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily VWAP for exit
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = np.divide(vwap_numerator, vwap_denominator, 
                     out=np.full_like(vwap_numerator, np.nan), 
                     where=vwap_denominator!=0)
    
    # Volume filter: 1.5x 20-day average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get weekly data for trend filter (EMA20)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_trend_up = close_1w > ema20_1w
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(vol_ma_period, 20)  # need enough data for VWAP and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(vwap[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(weekly_trend_up_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Calculate daily Camarilla pivot levels
        # Based on previous day's OHLC
        if i < 1:
            continue
            
        prev_high = high[i-1]
        prev_low = low[i-1]
        prev_close = close[i-1]
        range_val = prev_high - prev_low
        
        if range_val <= 0:
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
            
        # Camarilla levels
        s3 = prev_close - (range_val * 1.1 / 2)
        r3 = prev_close + (range_val * 1.1 / 2)
        
        if position == 1:  # Long position
            # Exit: price crosses below VWAP or weekly trend turns down
            if close[i] < vwap[i] or weekly_trend_up_aligned[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above VWAP or weekly trend turns up
            if close[i] > vwap[i] or weekly_trend_up_aligned[i] > 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches or goes below S3, volume surge, weekly trend up
            if (low[i] <= s3 and 
                vol_surge[i] and 
                weekly_trend_up_aligned[i] > 0.5):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches or goes above R3, volume surge, weekly trend down
            elif (high[i] >= r3 and 
                  vol_surge[i] and 
                  weekly_trend_up_aligned[i] < 0.5):
                position = -1
                signals[i] = -0.25
    
    return signals