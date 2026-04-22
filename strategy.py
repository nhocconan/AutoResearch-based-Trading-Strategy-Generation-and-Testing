#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla pivot reversal with 1d volume spike and 1w trend filter.
Long when price rejects S1/S2 with volume spike and 1w uptrend.
Short when price rejects R1/R2 with volume spike and 1w downtrend.
Exit when price crosses H/L or volume drops.
Uses 1w trend to avoid counter-trend trades in strong trends.
Designed for low trade frequency (20-40/year) to minimize fee drift.
"""
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
    
    # Load daily data for Camarilla levels and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Load weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # H, L, C from previous daily bar
    ph = df_daily['high'].shift(1).values
    pl = df_daily['low'].shift(1).values
    pc = df_daily['close'].shift(1).values
    
    # Camarilla levels
    r4 = pc + ((ph - pl) * 1.5000)
    r3 = pc + ((ph - pl) * 1.2500)
    r2 = pc + ((ph - pl) * 1.1666)
    r1 = pc + ((ph - pl) * 1.0833)
    s1 = pc - ((ph - pl) * 1.0833)
    s2 = pc - ((ph - pl) * 1.1666)
    s3 = pc - ((ph - pl) * 1.2500)
    s4 = pc - ((ph - pl) * 1.5000)
    
    # Align levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    r2_aligned = align_htf_to_ltf(prices, df_daily, r2)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    s2_aligned = align_htf_to_ltf(prices, df_daily, s2)
    h_align = align_htf_to_ltf(prices, df_daily, (ph + pl) / 2)  # midpoint for exit
    
    # Calculate 1d volume average (20-period)
    vol_daily = df_daily['volume'].values
    vol_avg_daily = pd.Series(vol_daily).rolling(window=20, min_periods=20).mean().values
    vol_avg_aligned = align_htf_to_ltf(prices, df_daily, vol_avg_daily)
    
    # Calculate 1w EMA34 for trend filter
    close_w = df_weekly['close'].values
    ema34_w = pd.Series(close_w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_w_aligned = align_htf_to_ltf(prices, df_weekly, ema34_w)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # warmup for weekly EMA
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(h_align[i]) or np.isnan(vol_avg_aligned[i]) or
            np.isnan(ema34_w_aligned[i])):
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
            # Long: Price near S1/S2 with volume spike and 1w uptrend
            near_support = (low[i] <= s1_aligned[i] * 1.002 and low[i] >= s2_aligned[i] * 0.998) or \
                          (low[i] <= s2_aligned[i] * 1.002 and low[i] >= s2_aligned[i] * 0.998)
            volume_spike = volume[i] > 2.0 * vol_avg_aligned[i]
            uptrend = close[i] > ema34_w_aligned[i]
            
            if near_support and volume_spike and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Price near R1/R2 with volume spike and 1w downtrend
            elif (high[i] >= r1_aligned[i] * 0.998 and high[i] <= r2_aligned[i] * 1.002) or \
                 (high[i] >= r2_aligned[i] * 0.998 and high[i] <= r2_aligned[i] * 1.002):
                volume_spike = volume[i] > 2.0 * vol_avg_aligned[i]
                downtrend = close[i] < ema34_w_aligned[i]
                if volume_spike and downtrend:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses H/L midpoint OR volume drops
                if close[i] > h_align[i] or volume[i] < 0.5 * vol_avg_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses H/L midpoint OR volume drops
                if close[i] < h_align[i] or volume[i] < 0.5 * vol_avg_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_CamarillaReversal_1dVolume_1wTrend"
timeframe = "4h"
leverage = 1.0
#%%