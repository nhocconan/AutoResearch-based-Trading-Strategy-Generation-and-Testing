#!/usr/bin/env python3
"""
1d_camarilla_pivot_1w_trend_volume_v1
Hypothesis: Daily chart strategy using weekly timeframe for context. 
Uses weekly CCI to determine trend (bullish >100, bearish <-100) and 
daily Camarilla pivot levels (S3/R3 for mean reversion, S4/R4 for breakouts). 
Volume confirmation filters false signals. Works in both bull and bear markets 
by adapting to weekly trend regime: in uptrend, favor long mean reversion at S3 
and breakout at R4; in downtrend, favor short mean reversion at R3 and 
breakout at S4. Designed for low trade frequency (target: 10-30 trades/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_camarilla_pivot_1w_trend_volume_v1"
timeframe = "1d"
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
    
    # Weekly data for trend context
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly CCI for trend
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tp = (high_1w + low_1w + close_1w) / 3
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - sma_tp) / (0.015 * mad)
    
    # Daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily OHLC for pivot points
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align weekly CCI to daily timeframe
    cci_1d = align_htf_to_ltf(prices, df_1w, cci)
    
    # Align daily levels to daily timeframe (no change, but for consistency)
    r4_1d = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_1d = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_1d = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_1d = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # 20-period volume average on daily
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(cci_1d[i]) or np.isnan(r3_1d[i]) or np.isnan(s3_1d[i]) or 
            np.isnan(r4_1d[i]) or np.isnan(s4_1d[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below S3 (mean reversion fail) OR 
            # price breaks above R4 and weekly trend turns bearish (CCI < -100)
            if close[i] < s3_1d[i] or (close[i] > r4_1d[i] and cci_1d[i] < -100):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above R3 (mean reversion fail) OR
            # price breaks below S4 and weekly trend turns bullish (CCI > 100)
            if close[i] > r3_1d[i] or (close[i] < s4_1d[i] and cci_1d[i] > 100):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion longs at S3 in weekly uptrend (CCI > 100)
            if (close[i] <= s3_1d[i] and 
                vol_confirm and 
                cci_1d[i] > 100):
                position = 1
                signals[i] = 0.25
            # Mean reversion shorts at R3 in weekly downtrend (CCI < -100)
            elif (close[i] >= r3_1d[i] and 
                  vol_confirm and 
                  cci_1d[i] < -100):
                position = -1
                signals[i] = -0.25
            # Breakout longs at R4 in weekly uptrend
            elif (close[i] >= r4_1d[i] and 
                  vol_confirm and 
                  cci_1d[i] > 100):
                position = 1
                signals[i] = 0.25
            # Breakout shorts at S4 in weekly downtrend
            elif (close[i] <= s4_1d[i] and 
                  vol_confirm and 
                  cci_1d[i] < -100):
                position = -1
                signals[i] = -0.25
    
    return signals