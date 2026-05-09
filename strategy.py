#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_1dTrend_Volume"
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
    
    # Weekly pivot: use Monday's OHLC (week start)
    # Create weekly timeframe from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get weekly OHLC from daily data (week starts Monday)
    weekly_high = []
    weekly_low = []
    weekly_close = []
    weekly_open = []
    
    for i in range(len(df_1d)):
        if i == 0 or df_1d.index[i].weekday() == 0:  # Monday
            weekly_high.append(df_1d['high'].iloc[i])
            weekly_low.append(df_1d['low'].iloc[i])
            weekly_open.append(df_1d['open'].iloc[i])
        else:
            weekly_high[-1] = max(weekly_high[-1], df_1d['high'].iloc[i])
            weekly_low[-1] = min(weekly_low[-1], df_1d['low'].iloc[i])
        weekly_close[-1] = df_1d['close'].iloc[i]
    
    # Convert to arrays and align to 6h
    weekly_high = np.array(weekly_high)
    weekly_low = np.array(weekly_low)
    weekly_close = np.array(weekly_close)
    weekly_open = np.array(weekly_open)
    
    # Align weekly data to 6h timeframe
    weekly_high_6h = align_htf_to_ltf(prices, df_1d, weekly_high)
    weekly_low_6h = align_htf_to_ltf(prices, df_1d, weekly_low)
    weekly_close_6h = align_htf_to_ltf(prices, df_1d, weekly_close)
    weekly_open_6h = align_htf_to_ltf(prices, df_1d, weekly_open)
    
    # Calculate weekly pivot points
    weekly_range = weekly_high_6h - weekly_low_6h
    weekly_pivot = (weekly_high_6h + weekly_low_6h + weekly_close_6h) / 3.0
    
    # Weekly support/resistance levels
    r1 = 2 * weekly_pivot - weekly_low_6h
    s1 = 2 * weekly_pivot - weekly_high_6h
    r2 = weekly_pivot + weekly_range
    s2 = weekly_pivot - weekly_range
    r3 = weekly_high_6h + 2 * (weekly_pivot - weekly_low_6h)
    s3 = weekly_low_6h - 2 * (weekly_high_6h - weekly_pivot)
    
    # Daily trend: EMA34 on 1d
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: volume > 1.3x 20-period SMA
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > 1.3 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(r2[i]) or np.isnan(s2[i]) or
            np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price above weekly R2 with daily uptrend and volume
            if (price > r2[i] and 
                price > ema34_1d_aligned[i] and 
                vol_filter[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price below weekly S2 with daily downtrend and volume
            elif (price < s2[i] and 
                  price < ema34_1d_aligned[i] and 
                  vol_filter[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns to weekly pivot or loses volume
            if (price < weekly_pivot[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly pivot or loses volume
            if (price > weekly_pivot[i] or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals