# 1d_WeeklyTrend_Camarilla_R3S3_Breakout
# Strategy: Use weekly trend direction from EMA200, combined with daily Camarilla R3/S3 breakouts.
# Enter long when price breaks above R3 in weekly uptrend, short when breaks below S3 in weekly downtrend.
# Requires volume confirmation (>1.5x 20-day average). Exits when price returns to Pivot or volume drops.
# Designed for low trade frequency (target 20-50/year) to minimize fee drag and work in both bull/bear markets.
# Weekly trend filter ensures we only trade with the higher timeframe momentum.
# Camarilla levels provide precise entry/exit points based on intraday volatility.
# Volume confirmation avoids false breakouts.
# Timeframe: 1d, HTF: 1w for trend filter.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyTrend_Camarilla_R3S3_Breakout"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (EMA200)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 200:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels and volume average
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_weekly = df_weekly['close'].values
    ema200_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 200:
        ema200_weekly[199] = np.mean(close_weekly[:200])
        for i in range(200, len(close_weekly)):
            ema200_weekly[i] = (close_weekly[i] * 2 + ema200_weekly[i-1] * 198) / 200
    
    # Calculate daily Camarilla levels (R3, S3, Pivot)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Camarilla: Pivot = (H+L+C)/3, Range = H-L
    pivot = (high_daily + low_daily + close_daily) / 3.0
    range_hl = high_daily - low_daily
    r3 = pivot + range_hl * 1.1 / 2.0  # R3 = P + 1.1*(H-L)/2
    s3 = pivot - range_hl * 1.1 / 2.0  # S3 = P - 1.1*(H-L)/2
    
    # Calculate daily volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align weekly trend to daily timeframe
    ema200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema200_weekly)
    
    # Align daily indicators to daily timeframe (no shift needed as they're same TF)
    pivot_aligned = pivot.copy()
    r3_aligned = r3.copy()
    s3_aligned = s3.copy()
    vol_avg_20_daily_aligned = vol_avg_20_daily.copy()
    
    # Pre-compute session filter (08-20 UTC) - optional but good practice
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # warmup for weekly EMA200 and daily volume
    
    for i in range(start_idx, n):
        # Skip if outside trading session (optional)
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema200_weekly_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: follow weekly trend with Camarilla breakout + volume
            # Weekly uptrend: price above EMA200
            weekly_uptrend = close[i] > ema200_weekly_aligned[i]
            # Weekly downtrend: price below EMA200
            weekly_downtrend = close[i] < ema200_weekly_aligned[i]
            
            # Long when price breaks above R3 in weekly uptrend with volume
            long_condition = (
                weekly_uptrend and
                close[i] > r3_aligned[i] and
                vol_confirm
            )
            
            # Short when price breaks below S3 in weekly downtrend with volume
            short_condition = (
                weekly_downtrend and
                close[i] < s3_aligned[i] and
                vol_confirm
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Pivot or volume drops significantly
            if close[i] < pivot_aligned[i] or volume[i] < 0.5 * vol_avg_20_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Pivot or volume drops significantly
            if close[i] > pivot_aligned[i] or volume[i] < 0.5 * vol_avg_20_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals