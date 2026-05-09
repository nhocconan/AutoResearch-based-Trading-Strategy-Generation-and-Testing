#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_PivotBreakout_VolumeTrend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate pivot and support/resistance from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Pivot point (standard)
    PP = (prev_high + prev_low + prev_close) / 3
    R1 = 2 * PP - prev_low
    S1 = 2 * PP - prev_high
    R2 = PP + prev_range
    S2 = PP - prev_range
    
    # Align to 12h timeframe
    PP_12h = align_htf_to_ltf(prices, df_1d, PP)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    R2_12h = align_htf_to_ltf(prices, df_1d, R2)
    S2_12h = align_htf_to_ltf(prices, df_1d, S2)
    
    # Get weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA34 for trend direction
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema_12h = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    # Volume filter: above 1.3x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(PP_12h[i]) or np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or
            np.isnan(R2_12h[i]) or np.isnan(S2_12h[i]) or
            np.isnan(weekly_ema_12h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.3 * vol_ma[i]  # Volume confirmation
        
        # Session filter: 08-20 UTC (reduce noise trades)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if position == 0:
            # Long: price breaks above R1 with weekly uptrend
            if (close[i] > R1_12h[i] and 
                close[i] > weekly_ema_12h[i] and  # weekly uptrend
                vol_ok and 
                in_session):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with weekly downtrend
            elif (close[i] < S1_12h[i] and 
                  close[i] < weekly_ema_12h[i] and  # weekly downtrend
                  vol_ok and 
                  in_session):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below PP (mean reversion to pivot)
            if close[i] < PP_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above PP (mean reversion to pivot)
            if close[i] > PP_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals