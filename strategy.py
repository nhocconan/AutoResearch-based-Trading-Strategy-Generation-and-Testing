#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R with weekly trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions. Combined with weekly EMA200 trend
# and volume > 1.5x 20-period average to filter false signals. Designed to capture
# mean-reversion bounces in ranging markets and trend continuations in trending markets.
# Target: 20-40 trades/year (80-160 total over 4 years). Works in bull/bear via trend filter.

name = "12h_WilliamsR_WeeklyTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA200 trend
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend
    close_weekly = df_weekly['close'].values
    ema200_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 200:
        ema200_weekly[199] = np.mean(close_weekly[:200])
        for i in range(200, len(close_weekly)):
            ema200_weekly[i] = (close_weekly[i] * 2 + ema200_weekly[i-1] * 198) / 200
    
    # Calculate 14-period Williams %R
    willr = np.full(n, np.nan)
    if n >= 14:
        for i in range(14, n):
            highest_high = np.max(high[i-13:i+1])
            lowest_low = np.min(low[i-13:i+1])
            if highest_high != lowest_low:
                willr[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
    
    # Calculate 12h volume average for volume filter
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align weekly EMA200 to 12h timeframe
    ema200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema200_weekly)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20, 14)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if np.isnan(willr[i]) or np.isnan(ema200_weekly_aligned[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Find current weekly bar's EMA200 (last completed weekly bar)
        ema200_weekly_current = np.nan
        if not np.isnan(ema200_weekly_aligned[i]):
            idx_weekly = 0
            while idx_weekly < len(df_weekly) and df_weekly.iloc[idx_weekly]['open_time'] <= prices.iloc[i]['open_time']:
                idx_weekly += 1
            idx_weekly -= 1  # last completed weekly bar
            
            if idx_weekly >= 0:
                ema200_weekly_current = ema200_weekly_aligned[i]
        
        if np.isnan(ema200_weekly_current):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        price_above_ema = close[i] > ema200_weekly_current
        price_below_ema = close[i] < ema200_weekly_current
        vol_filter = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: Williams %R extremes with trend and volume filter
            # Long when oversold (< -80) in uptrend, short when overbought (> -20) in downtrend
            if willr[i] < -80 and price_above_ema and vol_filter:  # Oversold in uptrend
                signals[i] = 0.25
                position = 1
            elif willr[i] > -20 and price_below_ema and vol_filter:  # Overbought in downtrend
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns to neutral or trend fails or volume drops
            if willr[i] > -50 or not price_above_ema or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns to neutral or trend fails or volume drops
            if willr[i] < -50 or not price_below_ema or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals