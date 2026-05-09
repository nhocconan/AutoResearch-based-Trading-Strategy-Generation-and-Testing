# 6h_MonthlyPivot_WeeklyTrend_Volume
# Hypothesis: Combines monthly pivot points (long-term structure) with weekly trend filter and volume confirmation.
# Monthly pivots provide institutional-grade support/resistance that adapts to long-term volatility.
# Weekly trend filter ensures we trade with the higher timeframe momentum, avoiding counter-trend traps.
# Volume confirmation ensures breakouts have institutional participation.
# Works in both bull and bear markets: In bull markets, buy above monthly resistance with weekly uptrend.
# In bear markets, sell below monthly support with weekly downtrend. Volume filter avoids false breakouts.

name = "6h_MonthlyPivot_WeeklyTrend_Volume"
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
    
    # Get monthly data for pivot calculation (using 1M approximation via 4-week aggregation)
    # Since we don't have 1M data, we'll use weekly data to calculate monthly-like pivots
    # by taking the first day of each month approximation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 4:
        return np.zeros(n)
    
    # Approximate monthly high/low/close by taking 4-week periods
    # We'll use 4-week rolling window to simulate monthly data
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # Calculate 4-period (approx monthly) high, low, close
    if len(high_w) >= 4:
        # Use 4-week rolling window for monthly approximation
        monthly_high = np.full_like(high_w, np.nan)
        monthly_low = np.full_like(low_w, np.nan)
        monthly_close = np.full_like(close_w, np.nan)
        
        for i in range(3, len(high_w)):
            monthly_high[i] = np.max(high_w[i-3:i+1])
            monthly_low[i] = np.min(low_w[i-3:i+1])
            monthly_close[i] = close_w[i]  # Use last week's close as monthly close
        
        # Shift to use previous month's data for pivot calculation
        prev_monthly_high = np.full_like(monthly_high, np.nan)
        prev_monthly_low = np.full_like(monthly_low, np.nan)
        prev_monthly_close = np.full_like(monthly_close, np.nan)
        
        prev_monthly_high[4:] = monthly_high[:-4]
        prev_monthly_low[4:] = monthly_low[:-4]
        prev_monthly_close[4:] = monthly_close[:-4]
        
        # Calculate monthly pivot points using previous month's data
        ph = prev_monthly_high  # previous month high
        pl = prev_monthly_low   # previous month low
        pc = prev_monthly_close # previous month close
        
        # Only calculate where we have valid data
        valid_m = (~np.isnan(ph)) & (~np.isnan(pl)) & (~np.isnan(pc))
        if np.any(valid_m):
            rang = ph - pl
            # Monthly pivot levels (R1, S1 are primary, R2/S2 for stronger moves)
            pivot = (ph + pl + pc) / 3
            r1 = 2 * pivot - pl
            s1 = 2 * pivot - ph
            r2 = pivot + rang
            s2 = pivot - rang
            
            # Align to 6h timeframe
            r1_aligned = align_htf_to_ltf(prices, df_w, r1)
            s1_aligned = align_htf_to_ltf(prices, df_w, s1)
            r2_aligned = align_htf_to_ltf(prices, df_w, r2)
            s2_aligned = align_htf_to_ltf(prices, df_w, s2)
        else:
            r1_aligned = s1_aligned = r2_aligned = s2_aligned = np.full(n, np.nan)
    else:
        r1_aligned = s1_aligned = r2_aligned = s2_aligned = np.full(n, np.nan)
    
    # Weekly trend filter: EMA crossover on weekly data
    if len(close_w) >= 21:
        ema_fast_w = np.full_like(close_w, np.nan)
        ema_slow_w = np.full_like(close_w, np.nan)
        
        # Fast EMA (9-period)
        ema_fast_w[8] = np.mean(close_w[0:9])
        for i in range(9, len(close_w)):
            ema_fast_w[i] = (close_w[i] * 0.2) + (ema_fast_w[i-1] * 0.8)
        
        # Slow EMA (21-period)
        ema_slow_w[20] = np.mean(close_w[0:21])
        for i in range(21, len(close_w)):
            ema_slow_w[i] = (close_w[i] * 0.0909) + (ema_slow_w[i-1] * 0.9091)
        
        # Align EMAs to 6h timeframe
        ema_fast_w_aligned = align_htf_to_ltf(prices, df_w, ema_fast_w)
        ema_slow_w_aligned = align_htf_to_ltf(prices, df_w, ema_slow_w)
        
        # Weekly trend: 1 for uptrend, -1 for downtrend
        weekly_trend = np.full_like(close, np.nan)
        valid_ema = (~np.isnan(ema_fast_w_aligned)) & (~np.isnan(ema_slow_w_aligned))
        weekly_trend[valid_ema] = np.where(ema_fast_w_aligned[valid_ema] > ema_slow_w_aligned[valid_ema], 1, -1)
    else:
        ema_fast_w_aligned = ema_slow_w_aligned = weekly_trend = np.full(n, np.nan)
    
    # Volume spike filter: current volume / 24-period average volume (4 days of 6h bars)
    vol_ma = np.full_like(volume, np.nan)
    if len(volume) >= 24:
        vol_ma[23] = np.mean(volume[0:24])
        for i in range(24, len(volume)):
            vol_ma[i] = (vol_ma[i-1] * 23 + volume[i]) / 24
    
    volume_ratio = np.full_like(volume, np.nan)
    valid = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid] = volume[valid] / vol_ma[valid]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 25)  # Ensure volume MA and weekly data are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(weekly_trend[i]) or np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above R1 AND weekly uptrend AND volume spike
            if (close[i] > r1_aligned[i] and 
                weekly_trend[i] == 1 and 
                volume_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 AND weekly downtrend AND volume spike
            elif (close[i] < s1_aligned[i] and 
                  weekly_trend[i] == -1 and 
                  volume_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 OR weekly trend turns down
            if close[i] < s1_aligned[i] or weekly_trend[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 OR weekly trend turns up
            if close[i] > r1_aligned[i] or weekly_trend[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals