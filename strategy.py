#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_daily_trend_v1
Hypothesis: Weekly and daily trend alignment with Camarilla pivot reversals on 12-hour chart.
In weekly uptrend (price above weekly EMA200), go long on price rebound from daily S3/S4 with volume confirmation.
In weekly downtrend (price below weekly EMA200), go short on price rebound from daily R3/R4 with volume confirmation.
Daily EMA50 filter ensures alignment with intermediate trend. Designed for low-frequency, high-conviction trades
that work in both bull (weekly uptrend longs) and bear (weekly downtrend shorts) markets.
Target: 15-25 trades per year on 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_daily_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    ema200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla calculations
    range_1d = prev_high - prev_low
    camarilla_S3 = prev_close - (range_1d * 1.1 / 6)
    camarilla_S4 = prev_close - (range_1d * 1.1 / 4)
    camarilla_R3 = prev_close + (range_1d * 1.1 / 6)
    camarilla_R4 = prev_close + (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 12h timeframe
    S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(R4_aligned[i]) or
            np.isnan(ema200_1w_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Price relative to Camarilla levels (within 0.2%)
        near_S3_S4 = (close[i] <= S3_aligned[i] * 1.002) or (close[i] <= S4_aligned[i] * 1.002)
        near_R3_R4 = (close[i] >= R3_aligned[i] * 0.998) or (close[i] >= R4_aligned[i] * 0.998)
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema200_1w_aligned[i]
        weekly_downtrend = close[i] < ema200_1w_aligned[i]
        
        # Daily trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price moves below S4 or weekly trend turns bearish
            if close[i] < S4_aligned[i] or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above R4 or weekly trend turns bullish
            if close[i] > R4_aligned[i] or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: weekly uptrend, price near S3/S4 with volume confirmation and daily uptrend
            if weekly_uptrend and near_S3_S4 and vol_confirmed and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short: weekly downtrend, price near R3/R4 with volume confirmation and daily downtrend
            elif weekly_downtrend and near_R3_R4 and vol_confirmed and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals