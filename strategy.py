#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Daily OHLC for weekly pivot calculation (1d timeframe) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # === Weekly OHLC from daily data ===
    # Resample daily to weekly: week starts Sunday, use last day of week
    # We'll create weekly arrays by sampling every 7th day starting from index 6
    # This gives us weekly OHLC: Friday's close as weekly close, etc.
    n_1d = len(high_1d)
    weeks = n_1d // 7
    if weeks < 2:
        return np.zeros(n)
    
    # Create weekly arrays (using Friday as week end for simplicity)
    week_idx = np.arange(6, n_1d, 7)  # Friday indices
    if len(week_idx) < 2:
        return np.zeros(n)
    
    weekly_high = np.max(high_1d[week_idx.reshape(-1,1) + np.arange(7)], axis=1)[:weeks]
    weekly_low = np.min(low_1d[week_idx.reshape(-1,1) + np.arange(7)], axis=1)[:weeks]
    weekly_close = close_1d[week_idx][:weeks]
    
    # Calculate weekly pivot points
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    weekly_r1 = weekly_close + weekly_range * 1.1 / 12
    weekly_s1 = weekly_close - weekly_range * 1.1 / 12
    
    # === ATR for volatility filter (14-period on daily) ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_avg = pd.Series(atr_1d).rolling(window=50, min_periods=50).mean().values
    
    # Align weekly data to 12h timeframe
    weekly_r1_12h = align_htf_to_ltf(prices, df_1d, weekly_r1, additional_delay_bars=0)
    weekly_s1_12h = align_htf_to_ltf(prices, df_1d, weekly_s1, additional_delay_bars=0)
    atr_1d_avg_12h = align_htf_to_ltf(prices, df_1d, atr_1d_avg, additional_delay_bars=0)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # === Price distance from weekly pivot (avoid chop) ===
    mid_pivot = (weekly_r1_12h + weekly_s1_12h) / 2
    dist_from_pivot = np.abs(close - mid_pivot)
    avg_dist = pd.Series(dist_from_pivot).rolling(window=50, min_periods=50).mean().values
    too_close = dist_from_pivot < (0.5 * avg_dist)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_r1_12h[i]) or np.isnan(weekly_s1_12h[i]) or
            np.isnan(atr_1d_avg_12h[i]) or np.isnan(volume_spike[i]) or
            np.isnan(too_close[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_level = weekly_r1_12h[i]
        s1_level = weekly_s1_12h[i]
        atr_avg = atr_1d_avg_12h[i]
        vol_spike = volume_spike[i]
        too_close_to_pivot = too_close[i]
        
        # === EXIT LOGIC: Exit when price moves against pivot or volatility drops ===
        if position == 1:  # Long position
            # Exit when price drops below S1 or volatility drops significantly
            if price < s1_level or (i > 0 and atr_avg < atr_1d_avg_12h[i-1] * 0.7):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above R1 or volatility drops significantly
            if price > r1_level or (i > 0 and atr_avg < atr_1d_avg_12h[i-1] * 0.7):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above weekly R1 with volume spike, sufficient volatility, and not too close to pivot
            if price > r1_level and vol_spike and atr_avg > 0 and not too_close_to_pivot:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below weekly S1 with volume spike, sufficient volatility, and not too close to pivot
            elif price < s1_level and vol_spike and atr_avg > 0 and not too_close_to_pivot:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Weekly_Pivot_R1_S1_Breakout_Volume_ATRFilter_DistFilter"
timeframe = "12h"
leverage = 1.0