#!/usr/bin/env python3
"""
6h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 6-hour timeframe, use weekly Camarilla pivot levels (R4/S4, R3/S3) to identify institutional support/resistance. Go long when price breaks above R4 with volume confirmation and weekly trend is up (price > weekly EMA20). Go short when price breaks below S4 with volume confirmation and weekly trend is down (price < weekly EMA20). The weekly timeframe provides structural bias while 6h captures medium-term moves. Designed for 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla pivot and trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels (based on previous week)
    # Using typical pivot calculation: (H + L + C) / 3
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pivot point
    pp = (weekly_high + weekly_low + weekly_close) / 3.0
    # Camarilla levels
    r4 = pp + ((weekly_high - weekly_low) * 1.500)
    r3 = pp + ((weekly_high - weekly_low) * 1.250)
    s3 = pp - ((weekly_high - weekly_low) * 1.250)
    s4 = pp - ((weekly_high - weekly_low) * 1.500)
    
    # Weekly trend: EMA20
    weekly_close_series = pd.Series(weekly_close)
    weekly_ema20 = weekly_close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly data to 6h timeframe (shifted by 1 week for no look-ahead)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Volume filter: 24-period average (4 days on 6h)
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if data not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(weekly_ema20_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below R3 or weekly trend turns down
            if close[i] < r3_aligned[i] or close[i] < weekly_ema20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above S3 or weekly trend turns up
            if close[i] > s3_aligned[i] or close[i] > weekly_ema20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish breakout: price closes above R4 with weekly uptrend
                if close[i] > r4_aligned[i] and close[i] > weekly_ema20_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Bearish breakdown: price closes below S4 with weekly downtrend
                elif close[i] < s4_aligned[i] and close[i] < weekly_ema20_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals