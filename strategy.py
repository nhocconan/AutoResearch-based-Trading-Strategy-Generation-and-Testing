#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeFilter"
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
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's range)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla calculations
    range_hl = daily_high - daily_low
    # Resistance 1 = C + (H-L)*1.1/12
    r1 = daily_close + range_hl * 1.1 / 12
    # Support 1 = C - (H-L)*1.1/12
    s1 = daily_close - range_hl * 1.1 / 12
    
    # Align daily Camarilla to 4h timeframe
    r1_4h = align_htf_to_ltf(prices, df_1d, r1)
    s1_4h = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: spike above 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 20)  # Wait for EMA34 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_4h[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(r1_4h[i]) or np.isnan(s1_4h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Pre-compute hour for session filter (UTC 0-24, 4h bars less sensitive)
        hour = pd.DatetimeIndex(prices['open_time']).hour[i]
        # Session filter: trade during active hours (6-22 UTC)
        in_session = (6 <= hour <= 22)
        
        if position == 0:
            # Long: price above S1, 1d uptrend (price > EMA34), volume breakout
            if (close[i] > s1_4h[i] and 
                close[i] > ema_34_4h[i] and 
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price below R1, 1d downtrend (price < EMA34), volume breakdown
            elif (close[i] < r1_4h[i] and 
                  close[i] < ema_34_4h[i] and 
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below S1 or trend reversal
            if close[i] < s1_4h[i] or close[i] < ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above R1 or trend reversal
            if close[i] > r1_4h[i] or close[i] > ema_34_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals