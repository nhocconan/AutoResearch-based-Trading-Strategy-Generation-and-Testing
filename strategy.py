#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_trend_volume_v3
Hypothesis: Use Camarilla pivot levels from daily timeframe with a 4-hour EMA trend filter. 
In uptrends (price > EMA20), go long at S3 (support) and exit at R3 (resistance). 
In downtrends (price < EMA20), go short at R3 (resistance) and exit at S3 (support). 
Volume confirmation filters out low-probability signals. 
This mean-reversion approach works in both bull and bear markets by adapting to trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_trend_volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    # Daily EMA20 for trend filter
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    
    # Align daily levels to 4h timeframe
    r3_4h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_4h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema20_4h = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # 20-period volume average on 4h
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r3_4h[i]) or np.isnan(s3_4h[i]) or 
            np.isnan(ema20_4h[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S3 resistance (mean reversion complete)
            if close[i] >= s3_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches R3 support (mean reversion complete)
            if close[i] <= r3_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion long at S3 in uptrend (price > EMA)
            if (close[i] <= s3_4h[i] and 
                vol_confirm and 
                close[i] > ema20_4h[i]):
                position = 1
                signals[i] = 0.25
            # Mean reversion short at R3 in downtrend (price < EMA)
            elif (close[i] >= r3_4h[i] and 
                  vol_confirm and 
                  close[i] < ema20_4h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals