# 6h_WeeklyPivot_Continuation_TrendFilter_v1
# Weekly pivot levels with trend continuation on 6h timeframe
# Uses weekly pivot points to identify institutional support/resistance
# In uptrend: buy pullbacks to weekly S1/S2; in downtrend: sell rallies to weekly R1/R2
# Trend filter uses 6h EMA50 to avoid counter-trend trades
# Volume confirmation ensures institutional participation
# Designed to work in both bull and bear markets by following the trend

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
    
    # === Weekly high/low/close for pivot calculation ===
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    weekly_close = df_weekly['close'].values
    
    # Calculate weekly pivot points: P = (H + L + C)/3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Support and resistance levels
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    
    # === Align weekly pivot levels to 6h timeframe ===
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_weekly, weekly_pivot)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s1)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_s2)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_weekly, weekly_r2)
    
    # === 6h EMA50 for trend filter ===
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # === 6h Volume confirmation ===
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma_20 * 1.5)  # Volume > 1.5x 20-period average
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or 
            np.isnan(weekly_s2_aligned[i]) or 
            np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_r2_aligned[i]) or 
            np.isnan(ema_50[i]) or 
            np.isnan(vol_confirm[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine trend based on 6h EMA50
        uptrend = close[i] > ema_50[i]
        downtrend = close[i] < ema_50[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long in uptrend: price near weekly support with volume confirmation
            if uptrend:
                # Price within 0.5% of S1 or S2
                near_s1 = abs(close[i] - weekly_s1_aligned[i]) / weekly_s1_aligned[i] < 0.005
                near_s2 = abs(close[i] - weekly_s2_aligned[i]) / weekly_s2_aligned[i] < 0.005
                if (near_s1 or near_s2) and vol_confirm[i]:
                    signals[i] = 0.25
                    position = 1
                    continue
            # Short in downtrend: price near weekly resistance with volume confirmation
            elif downtrend:
                # Price within 0.5% of R1 or R2
                near_r1 = abs(close[i] - weekly_r1_aligned[i]) / weekly_r1_aligned[i] < 0.005
                near_r2 = abs(close[i] - weekly_r2_aligned[i]) / weekly_r2_aligned[i] < 0.005
                if (near_r1 or near_r2) and vol_confirm[i]:
                    signals[i] = -0.25
                    position = -1
                    continue
        
        # Exit logic
        elif position == 1:
            # Exit long: price reaches weekly pivot or trend changes
            if close[i] >= weekly_pivot_aligned[i] or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reaches weekly pivot or trend changes
            if close[i] <= weekly_pivot_aligned[i] or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Continuation_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0