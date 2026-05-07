#!/usr/bin/env python3
# 6h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Use Camarilla pivot levels from daily timeframe (R3/S3 for reversal, R4/S4 for breakout)
# combined with daily trend filter (EMA34) and volume confirmation to capture both reversal and breakout moves.
# Works in bull markets via breakouts and in bear markets via reversals at extreme levels.
# Targets 15-30 trades/year to avoid fee drag while maintaining edge.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate previous day's Camarilla pivot levels
    # Using previous day's data to avoid look-ahead
    pp = np.zeros(len(high_1d))  # Pivot Point
    r4 = np.zeros(len(high_1d))  # Resistance 4
    r3 = np.zeros(len(high_1d))  # Resistance 3
    s3 = np.zeros(len(high_1d))  # Support 3
    s4 = np.zeros(len(high_1d))  # Support 4
    
    for i in range(1, len(high_1d)):
        # Use previous day's OHLC
        high_prev = high_1d[i-1]
        low_prev = low_1d[i-1]
        close_prev = close_1d[i-1]
        
        # Calculate pivot point
        pp[i] = (high_prev + low_prev + close_prev) / 3.0
        
        # Calculate ranges
        range_prev = high_prev - low_prev
        
        # Camarilla levels
        r4[i] = pp[i] + range_prev * 1.1 / 2.0
        r3[i] = pp[i] + range_prev * 1.1 / 4.0
        s3[i] = pp[i] - range_prev * 1.1 / 4.0
        s4[i] = pp[i] - range_prev * 1.1 / 2.0
    
    # Calculate daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: volume > 24-period average (24*6h = 4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Align 1d indicators to 6h timeframe
    pp_6h = align_htf_to_ltf(prices, df_1d, pp)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    ema_34_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(pp_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(r4_6h[i]) or 
            np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(ema_34_6h[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            vol_ok = volume[i] > vol_ma[i]
            
            # Long reversal: price at S3 with bullish trend
            if (close[i] <= s3_6h[i] * 1.005 and  # Allow small buffer
                close[i] > s4_6h[i] and          # Above S4 to avoid extreme
                ema_34_6h[i] > close[i] * 0.98 and  # Uptrend filter (price near EMA)
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short reversal: price at R3 with bearish trend
            elif (close[i] >= r3_6h[i] * 0.995 and   # Allow small buffer
                  close[i] < r4_6h[i] and            # Below R4 to avoid extreme
                  ema_34_6h[i] < close[i] * 1.02 and # Downtrend filter
                  vol_ok):
                signals[i] = -0.25
                position = -1
            # Long breakout: price breaks R4 with bullish trend
            elif (close[i] > r4_6h[i] and 
                  ema_34_6h[i] > close[i] * 0.95 and  # Strong uptrend
                  vol_ok):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks S4 with bearish trend
            elif (close[i] < s4_6h[i] and 
                  ema_34_6h[i] < close[i] * 1.05 and  # Strong downtrend
                  vol_ok):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price reaches R3 (take profit) or breaks S4 (stop reversal)
            if (close[i] >= r3_6h[i] * 0.995 or  # Near R3 for profit
                close[i] < s4_6h[i]):            # Breakdown below S4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches S3 (take profit) or breaks R4 (stop reversal)
            if (close[i] <= s3_6h[i] * 1.005 or  # Near S3 for profit
                close[i] > r4_6h[i]):            # Breakout above R4
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals