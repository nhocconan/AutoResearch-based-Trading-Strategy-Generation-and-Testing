#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla Pivot Breakout with 1d Volume Spike and 1w Trend Filter.
In trending markets, price breaks above/below Camarilla R1/S1 with strong volume.
In ranging markets, price reverses at Camarilla H4/L4 with volume confirmation.
Weekly trend filter ensures we trade in the direction of higher timeframe momentum.
Volume spike confirms institutional participation.
Designed for low trade frequency (20-50/year) to minimize fee drag.
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
    
    # Load daily data for Camarilla pivots and volume - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 10:
        return np.zeros(n)
    
    # Load weekly data for trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Based on previous day's high, low, close
    phigh = df_daily['high'].shift(1).values
    plow = df_daily['low'].shift(1).values
    pclose = df_daily['close'].shift(1).values
    
    # Calculate Camarilla levels
    # R4 = close + ((high-low) * 1.5)
    # R3 = close + ((high-low) * 1.25)
    # R2 = close + ((high-low) * 1.166)
    # R1 = close + ((high-low) * 1.083)
    # PP = (high + low + close) / 3
    # S1 = close - ((high-low) * 1.083)
    # S2 = close - ((high-low) * 1.166)
    # S3 = close - ((high-low) * 1.25)
    # S4 = close - ((high-low) * 1.5)
    diff = phigh - plow
    r1 = pclose + diff * 1.083
    s1 = pclose - diff * 1.083
    h4 = pclose + diff * 1.166  # Same as R2
    l4 = pclose - diff * 1.166  # Same as S2
    
    # Calculate weekly EMA20 for trend filter
    ema20_weekly = pd.Series(df_weekly['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate daily volume average (20-period)
    vol_avg_20 = pd.Series(df_daily['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF data to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_daily, r1)
    s1_4h = align_htf_to_ltf(prices, df_daily, s1)
    h4_4h = align_htf_to_ltf(prices, df_daily, h4)
    l4_4h = align_htf_to_ltf(prices, df_daily, l4)
    vol_avg_20_4h = align_htf_to_ltf(prices, df_daily, vol_avg_20)
    ema20_weekly_4h = align_htf_to_ltf(prices, df_weekly, ema20_weekly)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(r1_4h[i]) or np.isnan(s1_4h[i]) or 
            np.isnan(h4_4h[i]) or np.isnan(l4_4h[i]) or 
            np.isnan(vol_avg_20_4h[i]) or np.isnan(ema20_weekly_4h[i])):
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
            # Long: Price breaks above R1 with volume spike and weekly uptrend
            if (close[i] > r1_4h[i] and 
                volume[i] > 2.0 * vol_avg_20_4h[i] and 
                close[i] > ema20_weekly_4h[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume spike and weekly downtrend
            elif (close[i] < s1_4h[i] and 
                  volume[i] > 2.0 * vol_avg_20_4h[i] and 
                  close[i] < ema20_weekly_4h[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to opposite pivot level
            if position == 1:
                if close[i] < s1_4h[i]:  # Reverse to S1
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > r1_4h[i]:  # Reverse to R1
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4H_Camarilla_Pivot_Breakout_1dVolume_1wTrend"
timeframe = "4h"
leverage = 1.0
#%%