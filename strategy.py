#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R with weekly trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. Weekly trend provides bias,
# volume spike confirms momentum. Works in bull/bear by following weekly trend while
# fading extremes. Target: 20-40 trades/year.

name = "6h_WilliamsR_WeeklyTrend_Volume"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 50:
        ema50_weekly[49] = np.mean(close_weekly[:50])
        for i in range(50, len(close_weekly)):
            ema50_weekly[i] = (close_weekly[i] * 2 + ema50_weekly[i-1] * 48) / 50
    
    # Calculate 6-day Williams %R (14-period)
    highest_high_14 = np.full(len(high), np.nan)
    lowest_low_14 = np.full(len(low), np.nan)
    williams_r = np.full(len(close), np.nan)
    
    if len(high) >= 14:
        for i in range(14, len(high)):
            highest_high_14[i] = np.max(high[i-13:i+1])
            lowest_low_14[i] = np.min(low[i-13:i+1])
            if highest_high_14[i] > lowest_low_14[i]:
                williams_r[i] = -100 * (highest_high_14[i] - close[i]) / (highest_high_14[i] - lowest_low_14[i])
            else:
                williams_r[i] = -50  # neutral
    
    # Calculate 6-day volume average (20-period)
    vol_avg_20 = np.full(len(volume), np.nan)
    if len(volume) >= 20:
        for i in range(20, len(volume)):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align weekly indicators to 6h timeframe
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(williams_r[i]) or
            np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: fade extremes in direction of weekly trend
            # Williams %R < -80 = oversold, > -20 = overbought
            oversold = williams_r[i] < -80
            overbought = williams_r[i] > -20
            
            # Long when oversold and weekly uptrend
            long_condition = (
                oversold and
                close[i] > ema50_weekly_aligned[i] and   # weekly uptrend
                vol_confirm
            )
            
            # Short when overbought and weekly downtrend
            short_condition = (
                overbought and
                close[i] < ema50_weekly_aligned[i] and   # weekly downtrend
                vol_confirm
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: overbought or trend reversal
            if williams_r[i] > -20 or close[i] < ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: oversold or trend reversal
            if williams_r[i] < -80 or close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals