#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
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
    
    # Get daily data for trend and Camarilla calculation
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from daily OHLC
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Camarilla formula: Range = High - Low
    range_d = high_d - low_d
    # Resistance levels
    r3_d = close_d + range_d * 1.1 / 4
    r4_d = close_d + range_d * 1.1 / 2
    # Support levels
    s3_d = close_d - range_d * 1.1 / 4
    s4_d = close_d - range_d * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_d_aligned = align_htf_to_ltf(prices, df_d, r3_d)
    r4_d_aligned = align_htf_to_ltf(prices, df_d, r4_d)
    s3_d_aligned = align_htf_to_ltf(prices, df_d, s3_d)
    s4_d_aligned = align_htf_to_ltf(prices, df_d, s4_d)
    
    # Daily trend: EMA(34) on close
    ema34_d = pd.Series(close_d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need enough data for EMA and volume MA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(r3_d_aligned[i]) or 
            np.isnan(r4_d_aligned[i]) or
            np.isnan(s3_d_aligned[i]) or
            np.isnan(s4_d_aligned[i]) or
            np.isnan(ema34_d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_d_aligned[i]
        r4_val = r4_d_aligned[i]
        s3_val = s3_d_aligned[i]
        s4_val = s4_d_aligned[i]
        ema34_val = ema34_d_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price above R3 + above daily EMA34 + volume filter
            if close[i] > r3_val and close[i] > ema34_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below S3 + below daily EMA34 + volume filter
            elif close[i] < s3_val and close[i] < ema34_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below S3 or below EMA34
            if close[i] < s3_val or close[i] < ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above R3 or above EMA34
            if close[i] > r3_val or close[i] > ema34_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals