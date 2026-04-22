# 12H_Camarilla_R1_S1_WeeklyTrend_Volume
# Hypothesis: 12h Camarilla R1/S1 breakout in direction of weekly price trend with volume confirmation.
# Weekly trend: price above/below weekly EMA50. Volume confirms breakout strength.
# Camarilla levels provide precise support/resistance. Weekly trend filters direction.
# Target: 12-37 trades/year (50-150 total over 4 years).

#!/usr/bin/env python3
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
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend
    close_weekly = df_weekly['close'].values
    weekly_ema50 = pd.Series(close_weekly).ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    # Load daily data for Camarilla calculation
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous daily bar
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    
    # Previous day's values
    prev_high = np.roll(high_daily, 1)
    prev_low = np.roll(low_daily, 1)
    prev_close = np.roll(close_daily, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla formulas
    R1 = prev_close + (prev_high - prev_low) * 1.0833
    S1 = prev_close - (prev_high - prev_low) * 1.0833
    
    # Align Camarilla levels to 12h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_daily, R1)
    S1_aligned = align_htf_to_ltf(prices, df_daily, S1)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(weekly_ema50_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 AND price > weekly EMA50 (bullish trend) with volume
            if (close[i] > R1_aligned[i] and 
                close[i] > weekly_ema50_aligned[i] and 
                volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 AND price < weekly EMA50 (bearish trend) with volume
            elif (close[i] < S1_aligned[i] and 
                  close[i] < weekly_ema50_aligned[i] and 
                  volume[i] > 1.5 * vol_avg_20[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to the opposite Camarilla level or weekly EMA50
            if position == 1:
                if close[i] < S1_aligned[i] or close[i] < weekly_ema50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > R1_aligned[i] or close[i] > weekly_ema50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_Camarilla_R1_S1_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0