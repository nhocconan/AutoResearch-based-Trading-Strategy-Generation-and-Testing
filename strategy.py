#!/usr/bin/env python3
"""
6h_WeeklyPivot_PriceChannel
Hypothesis: Uses weekly pivot points to determine trend direction, price channel (ATR-based) for entries,
and volume confirmation. Works in both bull and bear markets by fading at weekly pivot extremes
and breaking out in the direction of the weekly trend. Targets 15-30 trades/year with discrete sizing.
"""

name = "6h_WeeklyPivot_PriceChannel"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data for pivot points and trend
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 1:
        return np.zeros(n)

    # Calculate weekly pivot points (standard floor trader method)
    # Using previous week's high, low, close
    wh = df_w['high'].values
    wl = df_w['low'].values
    wc = df_w['close'].values
    
    # Pivot point and support/resistance levels
    pp = (wh + wl + wc) / 3.0
    r1 = 2 * pp - wl
    s1 = 2 * pp - wh
    r2 = pp + (wh - wl)
    s2 = pp - (wh - wl)
    r3 = wh + 2 * (pp - wl)
    s3 = wl - 2 * (wh - pp)
    
    # Align weekly levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_w, s3)
    
    # Weekly trend: based on price vs pivot point
    weekly_trend = np.where(wc > pp, 1, -1)  # 1 for uptrend, -1 for downtrend
    weekly_trend_aligned = align_htf_to_ltf(prices, df_w, weekly_trend)
    
    # ATR for price channel calculation (14-period)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Price channel: upper and lower bands (ATR multiplier)
    atr_mult = 1.5
    upper_channel = close + atr_mult * atr
    lower_channel = close - atr_mult * atr
    
    # Volume confirmation: 20-period average
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(30, n):  # Start from 30 for sufficient data
        # Get aligned values for current 6h bar
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        trend_val = weekly_trend_aligned[i]
        vol_avg_val = vol_avg_20[i]
        atr_val = atr[i]
        upper_chan = upper_channel[i]
        lower_chan = lower_channel[i]
        
        # Skip if any required data is NaN
        if (np.isnan(pp_val) or np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(r2_val) or np.isnan(s2_val) or np.isnan(r3_val) or 
            np.isnan(s3_val) or np.isnan(trend_val) or np.isnan(vol_avg_val) or
            np.isnan(atr_val) or np.isnan(upper_chan) or np.isnan(lower_chan)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Price above lower channel + weekly uptrend + volume surge
            # Enter near support levels in uptrend
            if (close[i] > lower_chan and 
                trend_val == 1 and 
                volume[i] > vol_avg_val * 1.8):
                # Prefer entries near weekly support levels
                if close[i] <= s1_val * 1.02 or close[i] <= s2_val * 1.02:
                    signals[i] = 0.25
                    position = 1
            # SHORT: Price below upper channel + weekly downtrend + volume surge
            # Enter near resistance levels in downtrend
            elif (close[i] < upper_chan and 
                  trend_val == -1 and 
                  volume[i] > vol_avg_val * 1.8):
                # Prefer entries near weekly resistance levels
                if close[i] >= r1_val * 0.98 or close[i] >= r2_val * 0.98:
                    signals[i] = -0.25
                    position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price below lower channel or weekly trend turns down
            if (close[i] < lower_chan or trend_val == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price above upper channel or weekly trend turns up
            if (close[i] > upper_chan or trend_val == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals