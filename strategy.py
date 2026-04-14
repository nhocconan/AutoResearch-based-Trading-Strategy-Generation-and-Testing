#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly Pivot Reversal with Volume Confirmation
# Uses weekly pivot points (calculated from prior week) to identify reversal zones
# Price rejection at weekly S1/R1 with volume confirmation suggests institutional interest
# Trend filter from daily EMA (50) avoids counter-trend trades in strong trends
# Works in bull/bear markets by fading extremes in ranging markets and following trends in trending markets
# Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for pivot points
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    weekly_open = df_weekly['open'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe (wait for weekly close)
    pivot_aligned = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_weekly, r1)
    s1_aligned = align_htf_to_ltf(prices, df_weekly, s1)
    r2_aligned = align_htf_to_ltf(prices, df_weekly, r2)
    s2_aligned = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Load daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    ema_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Volume spike detector (2x average volume over 20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema_daily_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: only trade counter to strong daily trend at extremes
        above_daily_ema = price > ema_daily_aligned[i]
        
        if position == 0:
            # Long setup: price at weekly S1 with volume rejection and not in strong uptrend
            if (price <= s1_aligned[i] * 1.005 and  # within 0.5% of S1
                volume_spike[i] and 
                not above_daily_ema):  # not in strong uptrend
                position = 1
                signals[i] = position_size
            # Short setup: price at weekly R1 with volume rejection and not in strong downtrend
            elif (price >= r1_aligned[i] * 0.995 and  # within 0.5% of R1
                  volume_spike[i] and 
                  above_daily_ema):  # in strong uptrend (fade the strength)
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches weekly pivot or weekly R1 (take profit)
            if price >= pivot_aligned[i] or price >= r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches weekly pivot or weekly S1 (take profit)
            if price <= pivot_aligned[i] or price <= s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_WeeklyPivot_Reversal_Volume"
timeframe = "6h"
leverage = 1.0