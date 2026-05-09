# 4h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume
# Uses Camarilla pivot levels from daily timeframe with trend filter and volume confirmation
# Works in both bull and bear markets by only taking trades in direction of daily trend
# Limited to ~20-50 trades per year to minimize fee drag

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_Pivot_R3S3_Breakout_1dTrend_Volume"
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
    
    # Get daily data for Camarilla pivots and trend filter
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (R3, S3) from daily OHLC
    high_d = df_d['high'].values
    low_d = df_d['low'].values
    close_d = df_d['close'].values
    
    # Camarilla pivot calculations
    pivot_d = (high_d + low_d + close_d) / 3.0
    range_d = high_d - low_d
    
    # R3 and S3 levels (most significant for breakouts)
    r3_d = close_d + range_d * 1.1 / 2
    s3_d = close_d - range_d * 1.1 / 2
    
    # Align daily Camarilla levels to 4h timeframe
    r3_d_aligned = align_htf_to_ltf(prices, df_d, r3_d)
    s3_d_aligned = align_htf_to_ltf(prices, df_d, s3_d)
    
    # Daily trend filter: EMA34 > EMA89 for uptrend, EMA34 < EMA89 for downtrend
    close_series = pd.Series(close_d)
    ema34_d = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema89_d = close_series.ewm(span=89, adjust=False, min_periods=89).mean().values
    ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)
    ema89_d_aligned = align_htf_to_ltf(prices, df_d, ema89_d)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 89, 20)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(r3_d_aligned[i]) or 
            np.isnan(s3_d_aligned[i]) or
            np.isnan(ema34_d_aligned[i]) or
            np.isnan(ema89_d_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        r3_val = r3_d_aligned[i]
        s3_val = s3_d_aligned[i]
        ema34_val = ema34_d_aligned[i]
        ema89_val = ema89_d_aligned[i]
        vol_filter = volume_filter[i]
        
        # Determine trend direction
        uptrend = ema34_val > ema89_val
        downtrend = ema34_val < ema89_val
        
        if position == 0:
            # Enter long: Price breaks above R3 + uptrend + volume filter
            if close[i] > r3_val and uptrend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below S3 + downtrend + volume filter
            elif close[i] < s3_val and downtrend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below S3 or trend changes to downtrend
            if close[i] < s3_val or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above R3 or trend changes to uptrend
            if close[i] > r3_val or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals