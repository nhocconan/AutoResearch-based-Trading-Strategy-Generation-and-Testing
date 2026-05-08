#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot Breakout with 1d EMA Trend Filter and Volume Spike
# Uses daily Camarilla pivot levels (R3/S3) for breakout entries in the direction of daily EMA34 trend.
# Volume spike (>2x 20-period average) confirms institutional participation.
# Designed to work in both bull and bear markets by following the daily trend direction.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within optimal range.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_Volume"
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
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (R3, S3)
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate pivot point (PP)
    pp_daily = (high_daily + low_daily + close_daily) / 3.0
    range_daily = high_daily - low_daily
    
    # Camarilla levels: R3 = PP + 1.1 * (High - Low), S3 = PP - 1.1 * (High - Low)
    r3_daily = pp_daily + 1.1 * range_daily
    s3_daily = pp_daily - 1.1 * range_daily
    
    # Calculate daily EMA34 for trend filter
    ema34_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 34:
        ema34_daily[33] = np.mean(close_daily[:34])
        for i in range(34, len(close_daily)):
            ema34_daily[i] = (close_daily[i] * 2 + ema34_daily[i-1] * 32) / 34
    
    # Calculate daily volume average for volume filter
    vol_daily = df_daily['volume'].values
    vol_avg_20_daily = np.full(len(vol_daily), np.nan)
    if len(vol_daily) >= 20:
        for i in range(20, len(vol_daily)):
            vol_avg_20_daily[i] = np.mean(vol_daily[i-20:i])
    
    # Align daily indicators to 4h timeframe
    r3_daily_aligned = align_htf_to_ltf(prices, df_daily, r3_daily)
    s3_daily_aligned = align_htf_to_ltf(prices, df_daily, s3_daily)
    ema34_daily_aligned = align_htf_to_ltf(prices, df_daily, ema34_daily)
    vol_avg_20_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_20_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(r3_daily_aligned[i]) or np.isnan(s3_daily_aligned[i]) or
            np.isnan(ema34_daily_aligned[i]) or np.isnan(vol_avg_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current daily volume > 2x 20-period average
        vol_filter = False
        if not np.isnan(vol_avg_20_daily_aligned[i]):
            # Find current daily bar's volume
            idx_daily = 0
            while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
                idx_daily += 1
            idx_daily -= 1  # last completed daily bar
            
            if idx_daily >= 0:
                vol_daily_current = df_daily.iloc[idx_daily]['volume']
                vol_filter = vol_daily_current > 2.0 * vol_avg_20_daily_aligned[i]
        
        if position == 0:
            # Look for entry: Camarilla breakout with volume and trend filter
            # Long when price breaks above R3 in bullish trend
            long_condition = (
                close[i] > r3_daily_aligned[i] and   # price breaks above R3
                close[i] > ema34_daily_aligned[i] and # price above EMA34 (bullish trend)
                vol_filter
            )
            
            # Short when price breaks below S3 in bearish trend
            short_condition = (
                close[i] < s3_daily_aligned[i] and   # price breaks below S3
                close[i] < ema34_daily_aligned[i] and # price below EMA34 (bearish trend)
                vol_filter
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns below EMA34 or breaks below S3 (reversal)
            if close[i] < ema34_daily_aligned[i] or close[i] < s3_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns above EMA34 or breaks above R3 (reversal)
            if close[i] > ema34_daily_aligned[i] or close[i] > r3_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals