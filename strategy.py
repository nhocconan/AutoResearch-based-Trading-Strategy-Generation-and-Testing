#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Camarilla_R3S3_Breakout_1dTrend_Volume_v2"
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
    
    # Get 1d data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = pd.Series(df_1d['close'].values)
    ema34_1d = close_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate 1d volume average (20-period)
    vol_1d = pd.Series(df_1d['volume'].values)
    vol_ma20_1d = vol_1d.rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Current volume for confirmation
    vol_series = pd.Series(volume)
    vol_ma20_current = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d Camarilla levels (previous day's range)
    close_1d_series = pd.Series(df_1d['close'].values)
    high_1d_series = pd.Series(df_1d['high'].values)
    low_1d_series = pd.Series(df_1d['low'].values)
    range_1d = high_1d_series - low_1d_series
    
    # Camarilla levels based on previous day
    camarilla_high = close_1d_series.shift(1) + range_1d.shift(1) * 1.1 / 12  # R3
    camarilla_low = close_1d_series.shift(1) - range_1d.shift(1) * 1.1 / 12   # S3
    camarilla_high_ext = close_1d_series.shift(1) + range_1d.shift(1) * 1.1 / 6  # R4
    camarilla_low_ext = close_1d_series.shift(1) - range_1d.shift(1) * 1.1 / 6   # S4
    
    camarilla_high_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high.values)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low.values)
    camarilla_high_ext_aligned = align_htf_to_ltf(prices, df_1d, camarilla_high_ext.values)
    camarilla_low_ext_aligned = align_htf_to_ltf(prices, df_1d, camarilla_low_ext.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i]) or 
            np.isnan(vol_ma20_current[i]) or np.isnan(camarilla_high_aligned[i]) or 
            np.isnan(camarilla_low_aligned[i]) or np.isnan(camarilla_high_ext_aligned[i]) or 
            np.isnan(camarilla_low_ext_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20_current[i]
        
        if position == 0:
            # Long: Breakout above R3 with volume and above 1d EMA trend
            if close[i] > camarilla_high_aligned[i] and vol_ok and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below S3 with volume and below 1d EMA trend
            elif close[i] < camarilla_low_aligned[i] and vol_ok and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below R3 or trend reversal
            if close[i] < camarilla_low_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above S3 or trend reversal
            if close[i] > camarilla_high_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals