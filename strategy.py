#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly EMA(34) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema34_1w = close_1w.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily volume average (20-period)
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Current volume for confirmation
    vol_series = pd.Series(volume)
    vol_ma20_current = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or 
            np.isnan(vol_ma20_current[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Camarilla levels from previous day's OHLC
        prev_idx = i - 1
        if prev_idx < 0:
            continue
            
        # Get previous day's data (already aligned to 12h bars)
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Find the index for previous day in 1d data
        day_idx = prev_idx // 2  # 2x 12h bars per day
        if day_idx >= len(high_1d) or day_idx < 0:
            continue
            
        ph = high_1d[day_idx]
        pl = low_1d[day_idx]
        pc = close_1d[day_idx]
        
        # Calculate Camarilla levels
        range_ = ph - pl
        if range_ <= 0:
            continue
            
        # Resistance levels
        r3 = pc + (range_ * 1.1000 / 4)
        r4 = pc + (range_ * 1.1000 / 2)
        # Support levels
        s3 = pc - (range_ * 1.1000 / 4)
        s4 = pc - (range_ * 1.1000 / 2)
        
        vol_ok = volume[i] > 1.5 * vol_ma20_current[i]
        
        if position == 0:
            # Long: Break above R3 with volume and above weekly EMA trend
            if close[i] > r3 and vol_ok and close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with volume and below weekly EMA trend
            elif close[i] < s3 and vol_ok and close[i] < ema34_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below S3 or trend reversal
            if close[i] < s3 or close[i] < ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above R3 or trend reversal
            if close[i] > r3 or close[i] > ema34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals