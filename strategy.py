#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla pivot breakout with 1d volume confirmation and 1w EMA trend filter.
Long when price breaks above R1 with volume surge and price above 1w EMA34.
Short when price breaks below S1 with volume surge and price below 1w EMA34.
Exit when price crosses the pivot (HLC/4) or volume drops below average.
Designed for low trade frequency (15-30/year) to minimize fee drag on 12h timeframe.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for pivot calculation - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Load weekly data for EMA trend filter - ONCE before loop
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 35:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # HLC/4 = (High + Low + Close) / 4
    hlc4 = (df_daily['high'] + df_daily['low'] + df_daily['close']) / 4.0
    # R1 = Close + (High - Low) * 1.1/12
    r1 = df_daily['close'] + (df_daily['high'] - df_daily['low']) * 1.1 / 12.0
    # S1 = Close - (High - Low) * 1.1/12
    s1 = df_daily['close'] - (df_daily['high'] - df_daily['low']) * 1.1 / 12.0
    
    # Calculate weekly EMA34 for trend filter
    weekly_close = df_weekly['close'].values
    ema34_weekly = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HLC/4, R1, S1 to 12h timeframe
    hlc4_aligned = align_htf_to_ltf(prices, df_daily, hlc4.values)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1.values)
    
    # Align weekly EMA34 to 12h timeframe
    ema34_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Calculate 12h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(hlc4_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(ema34_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
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
            # Long: Price breaks above R1 with volume surge and above weekly EMA34
            if (close[i] > r1_aligned[i] and 
                volume[i] > 2.0 * vol_avg_20[i] and  # Strong volume surge
                close[i] > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume surge and below weekly EMA34
            elif (close[i] < s1_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_20[i] and  # Strong volume surge
                  close[i] < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below HLC/4 (pivot) OR volume drops
                if close[i] < hlc4_aligned[i] or volume[i] < 0.5 * vol_avg_20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above HLC/4 (pivot) OR volume drops
                if close[i] > hlc4_aligned[i] or volume[i] < 0.5 * vol_avg_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1S1_Breakout_1wEMA34_Volume"
timeframe = "12h"
leverage = 1.0
#%%