#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_PivotHighLow_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume average for volume filter (20-period)
    vol_1d_series = pd.Series(df_1d['volume'])
    vol_ma_1d = vol_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Calculate 6h-period high/low for breakout levels (20-period = ~5 days)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    highest_20 = high_series.rolling(window=20, min_periods=20).max().values
    lowest_20 = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 50)  # Need enough data for breakout levels and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hh = highest_20[i]
        ll = lowest_20[i]
        trend = ema50_1d_aligned[i]
        vol_ma = vol_ma_1d_aligned[i]
        vol_ratio = volume[i] / vol_ma if vol_ma > 0 else 0
        
        if position == 0:
            # Enter long: break above 20-period high with volume and above trend
            if close[i] > hh and volume[i] > vol_ma * 1.5 and close[i] > trend:
                signals[i] = 0.25
                position = 1
            # Enter short: break below 20-period low with volume and below trend
            elif close[i] < ll and volume[i] > vol_ma * 1.5 and close[i] < trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below 20-period low (mean reversion)
            if close[i] < ll:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above 20-period high (mean reversion)
            if close[i] > hh:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals