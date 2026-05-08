#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d EMA filter and volume confirmation
# Uses daily Camarilla pivot levels (S1/S3/R1/R3) from the previous day, filtered by daily EMA50 trend and volume > 1.5x 20-period average.
# Designed to capture institutional-level breakouts in trending markets while avoiding false breakouts in ranging markets.
# Works in bull/bear via EMA trend filter. Target: 20-50 trades/year (80-200 total over 4 years).

name = "4h_Camarilla_Pivot_Breakout_1dEMA50_VolumeConfirm"
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
    
    # Get daily data for Camarilla pivot calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Calculate pivot and Camarilla levels for each day
    pivot = np.full(len(close_daily), np.nan)
    r1 = np.full(len(close_daily), np.nan)
    r3 = np.full(len(close_daily), np.nan)
    s1 = np.full(len(close_daily), np.nan)
    s3 = np.full(len(close_daily), np.nan)
    
    for i in range(len(close_daily)):
        if i == 0:
            # For first day, use same day's data (will be filtered by alignment)
            high_val = high_daily[i]
            low_val = low_daily[i]
            close_val = close_daily[i]
        else:
            # Use previous day's data for today's levels
            high_val = high_daily[i-1]
            low_val = low_daily[i-1]
            close_val = close_daily[i-1]
        
        if not (np.isnan(high_val) or np.isnan(low_val) or np.isnan(close_val)):
            pivot[i] = (high_val + low_val + close_val) / 3
            range_val = high_val - low_val
            r1[i] = pivot[i] + range_val * 1.1 / 12
            r3[i] = pivot[i] + range_val * 1.1 / 4
            s1[i] = pivot[i] - range_val * 1.1 / 12
            s3[i] = pivot[i] - range_val * 1.1 / 4
    
    # Calculate daily EMA50 for trend
    ema50_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 50:
        ema50_daily[49] = np.mean(close_daily[:50])
        for i in range(50, len(close_daily)):
            ema50_daily[i] = (close_daily[i] * 2 + ema50_daily[i-1] * 48) / 50
    
    # Calculate 4h volume average for volume filter
    vol_avg_20 = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align daily Camarilla levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    r3_aligned = align_htf_to_ltf(prices, df_daily, r3)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    s3_aligned = align_htf_to_ltf(prices, df_daily, s3)
    
    # Align daily EMA50 to 4h timeframe
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(ema50_daily_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current daily bar's Camarilla levels (last completed daily bar)
        r1_current = np.nan
        r3_current = np.nan
        s1_current = np.nan
        s3_current = np.nan
        if not np.isnan(r1_aligned[i]):
            idx_daily = 0
            while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
                idx_daily += 1
            idx_daily -= 1  # last completed daily bar
            
            if idx_daily >= 0:
                r1_current = r1_aligned[i]
                r3_current = r3_aligned[i]
                s1_current = s1_aligned[i]
                s3_current = s3_aligned[i]
        
        # Get current daily bar's EMA50 (last completed daily bar)
        ema50_daily_current = np.nan
        if not np.isnan(ema50_daily_aligned[i]):
            idx_daily = 0
            while idx_daily < len(df_daily) and df_daily.iloc[idx_daily]['open_time'] <= prices.iloc[i]['open_time']:
                idx_daily += 1
            idx_daily -= 1  # last completed daily bar
            
            if idx_daily >= 0:
                ema50_daily_current = ema50_daily_aligned[i]
        
        if (np.isnan(r1_current) or np.isnan(r3_current) or 
            np.isnan(s1_current) or np.isnan(s3_current) or
            np.isnan(ema50_daily_current)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        price_above_r1 = close[i] > r1_current
        price_above_r3 = close[i] > r3_current
        price_below_s1 = close[i] < s1_current
        price_below_s3 = close[i] < s3_current
        price_above_ema = close[i] > ema50_daily_current
        price_below_ema = close[i] < ema50_daily_current
        vol_filter = volume[i] > 1.5 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: Camarilla breakout with EMA trend filter and volume confirmation
            # Long when price breaks above R1 in uptrend, short when breaks below S1 in downtrend
            if price_above_r1 and price_above_ema and vol_filter:
                signals[i] = 0.25
                position = 1
            elif price_below_s1 and price_below_ema and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to S1 or trend fails or volume drops
            if close[i] < s1_current or not price_above_ema or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to R1 or trend fails or volume drops
            if close[i] > r1_current or not price_below_ema or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals